import argparse
import json
import logging
import random
import time
import warnings
from itertools import permutations

import torch
import torch.nn.functional as F
from pytorch_lightning.loggers import LightningLoggerBase
from pytorch_lightning.utilities.distributed import rank_zero_only
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from transformers import logging as transformer_log, AutoTokenizer

transformer_log.set_verbosity_error()
warnings.filterwarnings("ignore", category=UserWarning)
LABEL = {"部件故障": 1, "性能故障": 2, "检测工具": 3, "组成": 4}
ID2LABEL = {v: k for k, v in LABEL.items()}
TYPE = {"部件单元": 1, "性能表征": 2, "故障状态": 3, "检测工具": 4}
LABEL2TYPE = {"部件故障": (TYPE["部件单元"], TYPE["故障状态"]),
              "性能故障": (TYPE["性能表征"], TYPE["故障状态"]),
              "检测工具": (TYPE["检测工具"], TYPE["性能表征"]),
              "组成": (TYPE["部件单元"], TYPE["部件单元"])}

SPAN_MASK = None


def get_offset(offset_mapping):
    start, end = {}, {}
    for i, offset in enumerate(offset_mapping[1:-1], 1):
        start[offset[0]] = i
        end[offset[1]] = i + 1
    return start, end


def count_space(s: str):
    length = len(s)
    ls = s.lstrip()
    left_length = len(ls)
    left = length - left_length

    rs = s.rstrip()
    right_length = len(rs)
    right = length - right_length
    return left, right


def fix_pos(entity, offset=1):
    name, pos = entity['name'], entity['pos']
    left, right = count_space(name)
    return [pos[0] + offset + left, pos[1] + offset - right]


def data_process(args):
    global SPAN_MASK
    SPAN_MASK = torch.zeros(args.max_len, args.max_len, dtype=torch.bool)
    for i in range(args.max_len):
        SPAN_MASK[i][i: i + args.max_entity_len] = True

    with open(args.raw_data_path, encoding='utf-8') as f:
        raw = f.readlines()
    data = [json.loads(_) for _ in raw]

    all_token = set()
    new_data = []
    for item in data:
        if not item.get('spo_list', None):
            continue
        item['text'] = item['text'].lower()
        all_token.update(item['text'].lower())
        new_data.append(item)

    all_token = list(all_token)
    tokenizer = AutoTokenizer.from_pretrained(args.bert_path)
    add_token_num = tokenizer.add_tokens(all_token)
    args.new_token_num = len(tokenizer)
    print('增加的token数量： ', add_token_num)
    print('数据集大小： ', len(new_data))
    del raw, data, all_token, add_token_num, item, f

    cls_id = tokenizer.cls_token_id
    sep_id = tokenizer.sep_token_id
    res = []
    not_same_entity = []
    max_entity_num = 0
    max_entity_len = 0
    random.shuffle(new_data)
    split_index = int(args.ratio * len(new_data))  # [:split_index]为训练集，[split_index:]为验证集

    for cur_index, item in enumerate(new_data):
        text = item['text'][:args.max_len - 2]  # 文章最长为args.max_len-2， 因为还有cls， 和sep
        input_id = [cls_id] + [tokenizer.convert_tokens_to_ids(i) for i in text] + [sep_id]
        text_len = len(input_id)

        spo_dict = {}
        entity = set()
        for spo in item['spo_list']:
            r = LABEL[spo['relation']]
            ht_type = LABEL2TYPE[spo['relation']]

            h, t = fix_pos(spo['h']), fix_pos(spo['t'])
            if h[1] >= args.max_len - 2 or t[1] >= args.max_len - 2:
                continue

            token_name = ''.join(tokenizer.convert_ids_to_tokens(input_id[h[0]: h[1]]))
            if token_name != spo['h']['name'].strip().lower():
                not_same_entity.append((token_name, spo['h']['name']))
            token_name = ''.join(tokenizer.convert_ids_to_tokens(input_id[t[0]: t[1]]))
            if token_name != spo['t']['name'].strip().lower():
                not_same_entity.append((token_name, spo['t']['name']))

            h.append(ht_type[0])
            h = tuple(h)
            t.append(ht_type[1])
            t = tuple(t)

            entity.update((h, t))
            spo_dict[(h, t)] = r

        entity = list(entity)
        max_entity_num = max(len(entity), max_entity_num)
        max_entity_len = max(max(map(lambda x: x[1] - x[0], entity)), max_entity_len)
        entity.sort(key=lambda x: x[0])

        relations = torch.tensor([spo_dict.get((h, t), 0) for h, t in permutations(entity, 2)])

        entity_num = len(entity)
        # entity_target = torch.zeros(text_len)
        entity_start = torch.zeros(entity_num)
        entity_end = torch.zeros(entity_num)

        # 多头标注
        span_target = torch.zeros((text_len, text_len))
        entity_type = torch.zeros(entity_num)
        for i, e in enumerate(entity):
            # e: [start_pos, end_pos + 1, type_id]
            # entity_target[e[0]] = e[2] if entity_target[e[0]] == 0 else 5
            entity_end[i] = e[1] - 1
            entity_start[i] = e[0]

            span_target[e[0], e[1] - 1] = e[2]
            entity_type[i] = e[2]

        res.append(dict(input_id=torch.tensor(input_id),
                        entity_num=entity_num,
                        # entity_target=entity_target,
                        entity_start=entity_start,
                        entity_end=entity_end,
                        relations=relations,
                        span_target=span_target,
                        entity_type=entity_type))
    train_data, val_data = res[:split_index], res[split_index:]
    test_data = process_test_data(args.raw_test_data, tokenizer, args.max_len)

    print('可能不匹配的实体数量： ', len(not_same_entity))
    print('最大实体数量', max_entity_num)
    print('最大实体长度', max_entity_len)
    return train_data, val_data, test_data


def process_test_data(raw_path, tokenizer, max_len):
    with open(raw_path, encoding='utf-8') as f:
        raw = f.readlines()
    data = [json.loads(_) for _ in raw]
    cls_id = tokenizer.cls_token_id
    sep_id = tokenizer.sep_token_id
    res = []
    for item in data:
        text = item['text']
        input_id = [cls_id] + [tokenizer.convert_tokens_to_ids(i) for i in text] + [sep_id]
        assert len(input_id) - 2 == len(item['text']), print(item['ID'])
        if len(input_id) > max_len:
            sep = input_id[-1]
            input_id = input_id[:1023] + [sep]
        res.append(dict(input_id=torch.tensor(input_id),
                        ID=item['ID'],
                        text=item['text']))
    return res


def get_result(ner_result, re_result, batch):
    batch_size = len(re_result)
    result = []
    for i in range(batch_size):
        text = batch['text'][i]
        start = (ner_result[0][i] - 1).tolist()
        end = ner_result[1][i].tolist()
        entity = [[start[j], end[j]] for j in range(len(start))]
        relations = re_result[i].tolist()
        k = -1
        spo_list = []
        for h_pos, t_pos in permutations(entity, 2):
            k += 1
            relation = ID2LABEL.get(relations[k], None)
            if relation is None:
                continue
            h = dict(name=text[h_pos[0]: h_pos[1]],
                     pos=h_pos)
            t = dict(name=text[t_pos[0]: t_pos[1]],
                     pos=t_pos)

            spo_list.append(dict(h=h, t=t, relation=relation))

        result.append(dict(ID=batch['ID'][i],
                           text=text,
                           spo_list=spo_list))
    return result


class Data(Dataset):
    def __init__(self, data):
        super(Data, self).__init__()
        self.data = data

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)


def create_mask(batch, input_id):
    attention_mask = torch.zeros_like(input_id, dtype=torch.bool)
    for i, x in enumerate(batch):
        attention_mask[i, :len(x['input_id'])] = True
    return attention_mask.bool()


def get_span_target(batch, max_len):
    res = []
    all_mask = []
    for x in batch:
        pad_len = max_len - len(x['input_id'])
        target = x['span_target']
        mask = SPAN_MASK[:len(x['input_id']), :len(x['input_id'])]
        target = F.pad(target, (0, pad_len, 0, pad_len))
        mask = F.pad(mask, (0, pad_len, 0, pad_len))
        res.append(target)
        all_mask.append(mask)
    res = torch.stack(res, dim=0)
    all_mask = torch.stack(all_mask, dim=0)
    return res, all_mask


def get_batch(batch):
    input_id = pad_sequence([x["input_id"] for x in batch], batch_first=True)
    attention_mask = create_mask(batch, input_id)
    if 'text' not in batch[0]:
        # relations = pad_sequence([torch.from_numpy(x["relations"]) for x in batch], batch_first=True)
        relations = torch.cat([x["relations"] for x in batch], dim=-1)
        batch_size, max_len = input_id.shape

        entity_nums = [x['entity_num'] for x in batch]
        entity_mask = torch.zeros(batch_size, max(entity_nums))
        for i in range(batch_size):
            entity_mask[i, :entity_nums[i]] = 1

        # entity_target = pad_sequence([x['entity_target'] for x in batch], batch_first=True)
        # entity_end = pad_sequence([x['entity_end'].long() for x in batch], batch_first=True)
        entity_end = [x['entity_end'].long() for x in batch]
        entity_start = [x['entity_start'].long() for x in batch]
        span_target, span_mask = get_span_target(batch, max_len)
        entity_type = [x['entity_type'].long() for x in batch]

        return dict(input_id=input_id,
                    # entity_target=entity_target.long(),
                    span_target=span_target.long(),
                    span_mask=span_mask,
                    entity_type=entity_type,
                    entity_start=entity_start,
                    entity_end=entity_end,
                    attention_mask=attention_mask,
                    relations=relations.long(),
                    entity_nums=entity_nums)
    else:
        text = [x['text'] for x in batch]
        text_id = [x['ID'] for x in batch]
        return dict(input_id=input_id,
                    attention_mask=attention_mask,
                    text=text,
                    ID=text_id)


class MyLogger(LightningLoggerBase):
    def __init__(self, args):
        super(MyLogger, self).__init__()
        logger = logging.getLogger('my_log')
        logger.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(message)s')

        fh = logging.FileHandler('log/' + args.save_name + '.txt')
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

        if args.print_log:
            ch = logging.StreamHandler()
            ch.setLevel(logging.DEBUG)
            ch.setFormatter(formatter)
            logger.addHandler(ch)
        self.logger = logger

    @property
    def name(self):
        return "MyLogger"

    @property
    def version(self):
        return time.strftime('%m.%d_%H:%M')

    @rank_zero_only
    def log_hyperparams(self, params):
        self.logger.info('=' * 32 + 'config information' + '=' * 32)
        args = {}
        for k in params.keys():
            if isinstance(params[k], argparse.Namespace):
                args.update(params[k].__dict__)
            else:
                args[k] = params[k]
        i = 1
        message = []
        for k, v in args.items():
            if isinstance(v, list) or isinstance(v, bool):
                continue
            message.append(f'{k}: {v}')
            if i % 2 == 0:
                message.append('\n')
            i += 1
        max_len = max(map(len, message))
        message = [m if m == '\n' else m + (max_len - len(m)) * ' ' for m in message]
        self.logger.info('|' + '|'.join(message))

    @rank_zero_only
    def log_metrics(self, metrics, step):
        epoch = metrics.pop('epoch')
        info = [f'{k}: {v: 5f}' if isinstance(v, float) and k != 'epoch' else f'{k}: {v}' for k, v in metrics.items()]
        info = ' | '.join(info)
        info = f'epoch: {epoch + 1: 3d} | step: {step + 1: 6d} | ' + info
        self.logger.info(info)
