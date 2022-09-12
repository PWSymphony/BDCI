import argparse
import json
import logging
import pickle
import time
import warnings
from itertools import permutations

import numpy as np
import torch
from pytorch_lightning.loggers import LightningLoggerBase
from pytorch_lightning.utilities.distributed import rank_zero_only
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import BertTokenizerFast, logging as transformer_log

transformer_log.set_verbosity_error()
warnings.filterwarnings("ignore", category=UserWarning)
LABEL = {"部件故障": 1, "性能故障": 2, "检测工具": 3, "组成": 4}
TYPE = {"部件单元": 1, "性能表征": 2, "故障状态": 3, "检测工具": 4}
LABEL2TYPE = {"部件故障": [TYPE["部件单元"], TYPE["故障状态"]],
              "性能故障": [TYPE["性能表征"], TYPE["故障状态"]],
              "检测工具": [TYPE["检测工具"], TYPE["性能表征"]],
              "组成": [TYPE["部件单元"], TYPE["部件单元"]]}


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


def fix_pos(name, pos, offset=1):
    left, right = count_space(name)
    return [pos[0] + offset + left, pos[1] + offset - right]


# def data_process(args):
#     with open(args.raw_data_path, encoding='utf-8') as f:
#         raw = f.readlines()
#     data = [json.loads(_) for _ in raw]
#     tokenizer = BertTokenizerFast.from_pretrained(args.bert_path)
#     tokenizer.add_tokens(ADD_TOKEN)
#     args.new_token_num = len(tokenizer)
#     CLS_id = tokenizer.cls_token_id
#     SEP_id = tokenizer.sep_token_id
#
#     res = []
#     not_same_entity = []
#     for item in tqdm(data):
#         input_id = [CLS_id] + [tokenizer.convert_tokens_to_ids(i) for i in item['text']] + [SEP_id]
#         assert len(input_id) - 2 == len(item['text']), print(item['ID'])
#         if len(input_id) > args.max_len:
#             sep = input_id[-1]
#             input_id = input_id[:1023] + [sep]
#
#         spo_list = item['spo_list']
#         spo_dict = {}
#         entity = set()
#         for spo in spo_list:
#             r = LABEL[spo['relation']]
#             ht_type = LABEL2TYPE[spo['relation']]
#
#             h = fix_pos(spo['h']['name'], spo['h']['pos'])
#             token_name = ''.join(tokenizer.convert_ids_to_tokens(input_id[h[0]: h[1]]))
#             if token_name != spo['h']['name']:
#                 not_same_entity.append((token_name, spo['h']['name']))
#
#             h.append(ht_type[0])
#             h = tuple(h)
#
#             t = fix_pos(spo['t']['name'], spo['t']['pos'])
#             token_name = ''.join(tokenizer.convert_ids_to_tokens(input_id[t[0]: t[1]]))
#             if token_name != spo['t']['name']:
#                 not_same_entity.append((token_name, spo['t']['name']))
#
#             t.append(ht_type[1])
#             t = tuple(t)
#
#             if h[1] >= args.max_len - 2 or t[1] >= args.max_len - 2:
#                 continue
#
#             entity.update((h, t))
#             spo_dict[(h, t)] = r
#
#         entity = list(entity)
#         entity.sort(key=lambda x: x[1])
#
#         relations = [spo_dict.get((h, t), 0) for h, t in permutations(entity, 2)]
#         entity_num = len(entity)
#         entity_pos_set = set()
#         entity_head_true = []
#         entity_tail_true = []
#         entity_type_true = []
#         for e in entity:
#             # e: [start_pos, end_pos + 1, type_id]
#             entity_pos_set.add((e[0], e[1] - 1))
#             entity_head_true.append(e[0])
#             entity_tail_true.append(e[1] - 1)
#             entity_type_true.append(e[2])
#
#         entity_head_false = []
#         entity_tail_false = []
#         for cur_len in range(1, args.span):
#             for start in range(1, len(input_id) - cur_len):
#                 end = start + cur_len - 1
#                 if (start, end) in entity_pos_set:
#                     continue
#                 entity_head_false.append(start)
#                 entity_tail_false.append(end)
#
#         res.append(dict(input_id=np.asarray(input_id),
#                         entity_num=entity_num,
#                         entity_head_true=np.asarray(entity_head_true),
#                         entity_tail_true=np.asarray(entity_tail_true),
#                         entity_type_true=np.asarray(entity_type_true),
#                         entity_head_false=np.asarray(entity_head_false),
#                         entity_tail_false=np.asarray(entity_tail_false),
#                         relations=np.asarray(relations),
#                         total_sample=args.total_sample - entity_num))
#     with open(args.data_path, 'wb') as f:
#         pickle.dump(res, f)
#     print('可能不匹配的实体数量： ', len(not_same_entity))

def data_process(args):
    with open(args.raw_data_path, encoding='utf-8') as f:
        raw = f.readlines()
    data = [json.loads(_) for _ in raw]
    all_token = set()
    for d in data:
        all_token.update(d['text'])
    all_token = list(all_token)
    tokenizer = BertTokenizerFast.from_pretrained(args.bert_path)
    add_token_num = tokenizer.add_tokens(all_token)
    args.new_token_num = len(tokenizer)
    CLS_id = tokenizer.cls_token_id
    SEP_id = tokenizer.sep_token_id

    res = []
    not_same_entity = []
    for item in tqdm(data):
        input_id = [CLS_id] + [tokenizer.convert_tokens_to_ids(i) for i in item['text']] + [SEP_id]
        assert len(input_id) - 2 == len(item['text']), print(item['ID'])
        if len(input_id) > args.max_len:
            sep = input_id[-1]
            input_id = input_id[:1023] + [sep]

        spo_list = item['spo_list']
        if not spo_list:
            continue
        spo_dict = {}
        entity = set()
        for spo in spo_list:
            r = LABEL[spo['relation']]
            ht_type = LABEL2TYPE[spo['relation']]

            h = fix_pos(spo['h']['name'], spo['h']['pos'])
            t = fix_pos(spo['t']['name'], spo['t']['pos'])
            if h[1] >= args.max_len - 2 or t[1] >= args.max_len - 2:
                continue

            token_name = ''.join(tokenizer.convert_ids_to_tokens(input_id[h[0]: h[1]]))
            if token_name != spo['h']['name'].strip():
                not_same_entity.append((token_name, spo['h']['name']))
            token_name = ''.join(tokenizer.convert_ids_to_tokens(input_id[t[0]: t[1]]))
            if token_name != spo['t']['name'].strip():
                not_same_entity.append((token_name, spo['t']['name']))

            h.append(ht_type[0])
            h = tuple(h)
            t.append(ht_type[1])
            t = tuple(t)

            entity.update((h, t))
            spo_dict[(h, t)] = r

        entity = list(entity)
        entity.sort(key=lambda x: x[1])

        relations = [spo_dict.get((h, t), 0) for h, t in permutations(entity, 2)]
        entity_num = len(entity)
        entity_target = np.zeros((len(input_id)))
        entity_start = np.zeros((len(entity)))
        entity_end = np.zeros((len(entity)))
        for i, e in enumerate(entity):
            # e: [start_pos, end_pos + 1, type_id]
            entity_target[e[0]] = e[2] if entity_target[e[0]] == 0 else 5
            entity_end[i] = e[1] - 1
            entity_start[i] = e[0]

        res.append(dict(input_id=np.asarray(input_id),
                        entity_num=entity_num,
                        entity_target=entity_target,
                        entity_start=entity_start,
                        entity_end=entity_end,
                        relations=np.asarray(relations),
                        total_sample=args.total_sample - entity_num))
    with open(args.data_path, 'wb') as f:
        pickle.dump(res, f)
    print('可能不匹配的实体数量： ', len(not_same_entity))
    print('增加的token数量： ', add_token_num)


class Data(Dataset):
    def __init__(self, path):
        super(Data, self).__init__()
        with open(path, 'rb') as f:
            self.data = pickle.load(f)

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)


def create_mask(batch, input_id):
    attention_mask = torch.zeros_like(input_id, dtype=torch.bool)
    for i, x in enumerate(batch):
        attention_mask[i, :len(x['input_id'])] = True
    return attention_mask


def get_sample(batch, is_test):
    entity_heads = []
    entity_tails = []
    entity_labels = []
    for x in batch:
        if is_test:
            entity_head = np.concatenate((x['entity_head_true'], x['entity_head_false']), axis=0)
            entity_tail = np.concatenate((x['entity_tail_true'], x['entity_tail_false']), axis=0)
            entity_label = np.zeros_like(entity_head)
            entity_label[:x['entity_num']] = x['entity_type_true']
        else:
            index = np.random.choice(len(x['entity_head_false']), x['total_sample'])
            entity_head = np.concatenate((x['entity_head_true'], x['entity_head_false'][index]), axis=0)
            entity_tail = np.concatenate((x['entity_tail_true'], x['entity_tail_false'][index]), axis=0)
            entity_label = np.concatenate((x['entity_type_true'], np.zeros((x['total_sample']))), axis=0)

            index = np.arange(0, len(entity_head))
            np.random.shuffle(index)
            entity_head = entity_head[index]
            entity_tail = entity_tail[index]
            entity_label = entity_label[index]

        entity_heads.append(entity_head)
        entity_tails.append(entity_tail)
        # entity_labels.append(np.eye(len(LABEL) + 1)[list(entity_label.astype(int))])
        entity_labels.append(entity_label)

    entity_heads = np.stack(entity_heads, axis=0)
    entity_tails = np.stack(entity_tails, axis=0)
    entity_labels = torch.from_numpy(np.stack(entity_labels, axis=0))
    spans = torch.from_numpy(entity_tails - entity_heads + 1)

    return entity_heads, entity_tails, entity_labels, spans


def get_batch(batch, is_test=False):
    input_id = pad_sequence([torch.from_numpy(x["input_id"]) for x in batch], batch_first=True)
    relations = pad_sequence([torch.from_numpy(x["relations"]) for x in batch], batch_first=True)
    attention_mask = create_mask(batch, input_id)
    batch_size, max_len = input_id.shape

    entity_nums = [x['entity_num'] for x in batch]
    entity_mask = torch.zeros(batch_size, max(entity_nums))
    for i in range(batch_size):
        entity_mask[i, :entity_nums[i]] = 1
    # entity_heads, entity_tails, entity_labels, spans = get_sample(batch, is_test)

    entity_target = pad_sequence([torch.from_numpy(x['entity_target']) for x in batch], batch_first=True)
    entity_end = pad_sequence([torch.from_numpy(x['entity_end']) for x in batch], batch_first=True)
    entity_start = [torch.from_numpy(x['entity_start']).long() for x in batch]

    return dict(input_id=input_id,
                entity_target=entity_target.long(),
                entity_start=entity_start,
                entity_end=entity_end.long(),
                attention_mask=attention_mask.bool(),
                relations=relations.long(),
                entity_mask=entity_mask.bool())


# def get_batch_single(batch):
#     input_id = torch.from_numpy(batch[0]['input_id']).unsqueeze(0)
#     attention_mask = create_mask(batch, input_id)
#     entity_heads = np.expand_dims(batch[0]['entity_head_false'], axis=0)
#     entity_tails = np.expand_dims(batch[0]['entity_tail_false'], axis=0)
#     spans = torch.from_numpy(entity_tails - entity_heads + 1)
#
#     res = dict(input_id=input_id,
#                entity_heads=entity_heads,
#                entity_tails=entity_tails,
#                spans=spans,
#                attention_mask=attention_mask)
#
#     if 'relations' in batch[0] and batch[0]['relations'] is not None:
#         relations = np.expand_dims(batch[0]['relations'], axis=0)
#         entity_label = np.zeros_like(entity_heads)
#
#     return res


class MyLogger(LightningLoggerBase):
    def __init__(self, args):
        super(MyLogger, self).__init__()
        logger = logging.getLogger('my_log')
        logger.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(message)s')

        fh = logging.FileHandler(args.save_name + '.txt')
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
