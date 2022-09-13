import argparse
from itertools import permutations, accumulate

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from transformers import BertModel

from loss import CELoss


def process_long_input(model, input_ids, attention_mask, start_tokens, end_tokens):
    # Split the input to 2 overlapping chunks. Now BERT can encode inputs of which the length are up to 1024.
    # 例如输入长度最大为5，当输入长度为7：[1,2,3,4,5,6,7]时，分成[1,2,3,4,5]和[3,4,5,6,7]两部分

    n, c = input_ids.size()
    start_tokens = torch.tensor(start_tokens).to(input_ids)
    end_tokens = torch.tensor(end_tokens).to(input_ids)
    len_start = start_tokens.size(0)
    len_end = end_tokens.size(0)
    if c <= 512:
        output = model(input_ids=input_ids,
                       attention_mask=attention_mask,
                       output_attentions=True)
        sequence_output = output[0]
        attention = output[-1][-1]
    else:
        new_input_ids, new_attention_mask, num_seg = [], [], []
        seq_len = attention_mask.sum(1).cpu().numpy().astype(np.int32).tolist()
        for i, l_i in enumerate(seq_len):
            if l_i <= 512:
                new_input_ids.append(input_ids[i, :512])
                new_attention_mask.append(attention_mask[i, :512])
                num_seg.append(1)
            else:
                input_ids1 = torch.cat([input_ids[i, :512 - len_end], end_tokens], dim=-1)
                input_ids2 = torch.cat([start_tokens, input_ids[i, (l_i - 512 + len_start): l_i]], dim=-1)
                attention_mask1 = attention_mask[i, :512]
                attention_mask2 = attention_mask[i, (l_i - 512): l_i]
                new_input_ids.extend([input_ids1, input_ids2])
                new_attention_mask.extend([attention_mask1, attention_mask2])
                num_seg.append(2)
        input_ids = torch.stack(new_input_ids, dim=0)
        attention_mask = torch.stack(new_attention_mask, dim=0)
        output = model(input_ids=input_ids,
                       attention_mask=attention_mask,
                       output_attentions=True)
        sequence_output = output[0]
        attention = output[-1][-1]
        i = 0
        new_output, new_attention = [], []
        for (n_s, l_i) in zip(num_seg, seq_len):
            if n_s == 1:
                output = F.pad(sequence_output[i], (0, 0, 0, c - 512))
                att = F.pad(attention[i], (0, c - 512, 0, c - 512))
                new_output.append(output)
                new_attention.append(att)
            elif n_s == 2:
                output1 = sequence_output[i][:512 - len_end]
                mask1 = attention_mask[i][:512 - len_end]
                att1 = attention[i][:, :512 - len_end, :512 - len_end]
                output1 = F.pad(output1, (0, 0, 0, c - 512 + len_end))
                mask1 = F.pad(mask1, (0, c - 512 + len_end))
                att1 = F.pad(att1, (0, c - 512 + len_end, 0, c - 512 + len_end))

                output2 = sequence_output[i + 1][len_start:]
                mask2 = attention_mask[i + 1][len_start:]
                att2 = attention[i + 1][:, len_start:, len_start:]
                output2 = F.pad(output2, (0, 0, l_i - 512 + len_start, c - l_i))
                mask2 = F.pad(mask2, (l_i - 512 + len_start, c - l_i))
                att2 = F.pad(att2, [l_i - 512 + len_start, c - l_i, l_i - 512 + len_start, c - l_i])
                mask = mask1 + mask2 + 1e-10
                output = (output1 + output2) / mask.unsqueeze(-1)
                att = (att1 + att2)
                att = att / (att.sum(-1, keepdim=True) + 1e-10)  # 归一化
                new_output.append(output)
                new_attention.append(att)
            i += n_s
        sequence_output = torch.stack(new_output, dim=0)
        attention = torch.stack(new_attention, dim=0)
    return sequence_output, attention


def compute_score(batch, ner_result, re_result):
    all_true_start = batch['entity_start']
    true_entity_num = [len(x) for x in all_true_start]
    true_relations = [x * (x - 1) for x in true_entity_num]
    true_relations = [0] + list(accumulate(true_relations))
    pred_entity_num = [len(x) for x in ner_result[0]]

    ner_true_num, ner_recall_num, ner_pred_num = 0, sum(true_entity_num), sum(pred_entity_num)
    re_true_num, re_recall_num, re_pred_num = 0, float(batch['relations'].bool().sum()), 0

    for i in range(len(ner_result[0])):
        if not re_result[i].tolist():
            continue
        true_start = all_true_start[i]
        true_e_num = true_entity_num[i]
        true_relation = batch['relations'][true_relations[i]: true_relations[i + 1]].tolist()

        cur_re_result = re_result[i].tolist()
        cur_result = torch.stack((ner_result[0][i], ner_result[1][i])).permute(1, 0).tolist()
        ner_pred = [tuple(x) for x in cur_result]

        j = -1
        re_pred = []
        for h, t in permutations(ner_pred, 2):
            j += 1
            if cur_re_result[j] != 0:
                re_pred.append((h[0], h[1], t[0], t[1], cur_re_result[j]))
        re_pred_num += len(re_pred)

        ner_true = torch.stack((true_start, batch['entity_end'][i, :true_e_num])).permute(1, 0).tolist()
        ner_true = [tuple(x) for x in ner_true]

        j = -1
        re_true = []
        for h, t in permutations(ner_true, 2):
            j += 1
            if true_relation[j] != 0:
                re_true.append((h[0], h[1], t[0], t[1], true_relation[j]))

        ner_pred, ner_true = set(ner_pred), set(ner_true)
        ner_true_num += len(ner_pred & ner_true)

        re_pred, re_true = set(re_pred), set(re_true)
        re_true_num += len(re_pred & re_true)

    return ner_true_num, ner_recall_num, ner_pred_num, re_true_num, re_recall_num, re_pred_num


class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            batch_first=True, bidirectional=True)
        self.in_dropout = nn.Dropout(dropout)

    def forward(self, src, src_lengths, pad_value=0):
        """
        src: [batch_size, len, input_size]
        src_lengths: [batch_size]
        """
        src = self.in_dropout(src)
        packed_src = nn.utils.rnn.pack_padded_sequence(src, src_lengths.cpu(), batch_first=True,
                                                       enforce_sorted=False)
        packed_outputs, _ = self.lstm(packed_src)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs, batch_first=True, padding_value=pad_value)
        return outputs


class GLU(nn.Module):
    def __init__(self, in_feature, out_feature):
        super(GLU, self).__init__()
        self.u_dense = nn.Linear(in_feature, 3 * in_feature)
        self.v_dense = nn.Linear(in_feature, 3 * in_feature)
        self.o_dense = nn.Linear(3 * in_feature, out_feature)

    def forward(self, src):
        u = self.u_dense(src)
        v = self.v_dense(src)
        uv = F.sigmoid(u) * v
        return self.o_dense(uv)


class FFNN(nn.Module):
    def __init__(self, in_feature, out_feature, mid_size=256, dropout=0.2):
        super(FFNN, self).__init__()
        self.dense0 = nn.Linear(in_feature, mid_size)
        self.dense1 = nn.Linear(mid_size, out_feature)
        self.drop = nn.Dropout(dropout)

    def forward(self, src):
        output = self.dense0(src)
        output = F.relu(output)
        output = self.drop(output)
        output = self.dense1(output)

        return output


class GroupLinear(nn.Module):
    def __init__(self, in_feature, out_feature, block_size=64):
        super(GroupLinear, self).__init__()
        assert in_feature % block_size == 0
        self.linear = nn.Linear(in_feature * block_size, out_feature)
        self.block_size = block_size

    def forward(self, input1, input2):
        l, h = input1.shape
        input1 = input1.reshape(l, h // self.block_size, self.block_size)
        input2 = input2.reshape(l, h // self.block_size, self.block_size)

        output = input1.unsqueeze(-1) * input2.unsqueeze(-2)
        output = output.reshape(l, h * self.block_size)

        return self.linear(output)


class NER(nn.Module):
    def __init__(self, hidden_size, args):
        super(NER, self).__init__()
        self.start_fc = GLU(hidden_size, args.tag_size)

        self.q_fc = nn.Linear(hidden_size, hidden_size)
        self.k_fc = nn.Linear(hidden_size, hidden_size)
        self.heads = 12
        self.scale = (hidden_size // 12) ** 0.5

        self.loss_fn = CELoss()

    def forward(self, hidden_state, batch, is_test=False):
        batch_size = hidden_state.shape[0]
        attention_mask = batch['attention_mask']
        start_logit = self.start_fc(hidden_state)

        if is_test:
            entity_index = torch.argmax(start_logit, dim=-1) * batch['attention_mask']
            text_len = batch['attention_mask'].sum(-1) - 1
            entity_index[:, 0] = 0
            for i in range(batch_size):
                entity_index[i, text_len[i]] = 0
            entity_num = entity_index.bool().sum(-1).tolist()

            entity_mask = torch.zeros((hidden_state.shape[0], max(entity_num)), dtype=torch.bool,
                                      device=hidden_state.device)

            for i in range(hidden_state.shape[0]):
                entity_mask[i, :entity_num[i]] = True

            entity_index = torch.nonzero(entity_index)
            entity = hidden_state[entity_index[:, 0], entity_index[:, 1]]
            entity = torch.split(entity, entity_num, dim=0)
            entity = pad_sequence(entity, batch_first=True)
        else:
            entity_index = None
            entity_num = [len(batch['entity_start'][i]) for i in range(hidden_state.shape[0])]
            entity = []
            for i, x in enumerate(batch['entity_start']):
                entity.append(hidden_state[i][x])
            entity = pad_sequence(entity, batch_first=True)

        q = self.q_fc(entity)
        q = q.reshape(q.shape[0], q.shape[1], self.heads, -1)
        q = q.permute(0, 2, 1, 3)

        k = self.k_fc(hidden_state)
        k = k.reshape(k.shape[0], k.shape[1], self.heads, -1)
        k = k.permute(0, 2, 3, 1)

        end_logit = (q @ k).mean(dim=1) / self.scale
        end_logit = end_logit.masked_fill(~(attention_mask.unsqueeze(1)), -5e4)
        if is_test:
            loss = None
            entity_start_index = torch.split(entity_index, entity_num, dim=0)
            entity_start_index = [x[:, 1] for x in entity_start_index]

            entity_end_index = end_logit.max(-1)[1]
            entity_end_index = [entity_end_index[i, :entity_num[i]] for i in range(hidden_state.shape[0])]
            new_entity_start_index, new_end_start_index = [], []
            for i in range(batch_size):
                delta = entity_end_index[i] - entity_start_index[i]
                index = (0 <= delta) & (delta < 20)
                new_entity_start_index.append(entity_start_index[i][index])
                new_end_start_index.append(entity_end_index[i][index])
            entity_start_index = new_entity_start_index
            entity_end_index = new_end_start_index
        else:
            entity_target = batch['entity_target']
            entity_end = batch['entity_end']

            loss1 = self.loss_fn(start_logit, entity_target, attention_mask)
            loss2 = self.loss_fn(end_logit, entity_end, batch['entity_mask'])
            loss = loss1 + loss2
            entity_start_index = batch['entity_start']

            entity_end_index = [entity_end[i, :entity_num[i]] for i in range(hidden_state.shape[0])]

        return loss, (entity_start_index, entity_end_index)


# class NER(nn.Module):
#     def __init__(self, hidden_size, args):
#         super(NER, self).__init__()
#         self.glu = GLU(hidden_size * 2 + args.ner_emb_dim, args.tag_size)
#         self.dropout = nn.Dropout(p=args.ner_dropout)
#         # self.dense = nn.Linear(hidden_size * 2 + args.ner_emb_dim, args.tag_size)
#         # self.dense = nn.Bilinear(hidden_size + args.ner_emb_dim, hidden_size + args.ner_emb_dim, args.tag_size)
#         # self.FFNN = FFNN(hidden_size * 2 + args.ner_emb_dim, args.tag_size)
#         self.span_emb = nn.Embedding(num_embeddings=args.span, embedding_dim=args.ner_emb_dim)
#         self.loss_fn = CELoss()
#
#     def forward(self, hidden_state, batch):
#         batch_size = hidden_state.shape[0]
#         entity_heads = batch['entity_heads']
#         entity_tails = batch['entity_tails']
#         spans = self.span_emb(batch['spans'])
#         entity_head_feature = torch.stack([hidden_state[i, entity_heads[i], :] for i in range(batch_size)], dim=0)
#         entity_tail_feature = torch.stack([hidden_state[i, entity_tails[i], :] for i in range(batch_size)], dim=0)
#         # 方案1：拼接头尾token特征, 再用glu分类
#         entity_feature = self.dropout(torch.cat((entity_head_feature, entity_tail_feature), dim=-1))
#         entity_feature = torch.cat((entity_feature, spans), dim=-1)
#         entity_feature = self.glu(entity_feature)
#
#         # 方案2：头尾实体分别拼接span特征，在用bilinear分类
#         # entity_head_feature = torch.cat((entity_head_feature, spans), dim=-1)
#         # entity_tail_feature = torch.cat((entity_tail_feature, spans), dim=-1)
#         # entity_feature = self.dense(entity_head_feature, entity_tail_feature)
#
#         # 方案3：头尾实体分别拼接span特征，再用前馈神经网络分类
#         # entity_feature = torch.cat((entity_head_feature, entity_tail_feature, spans), dim=-1)
#         # entity_feature = self.FFNN(entity_feature)
#
#         if 'entity_labels' in batch:
#             loss = self.loss_fn(entity_feature, batch['entity_labels'])
#             return loss, self.loss_fn.get_label(entity_feature)
#         else:
#             return None, self.loss_fn.get_label(entity_feature)

class ExtraRelation(nn.Module):
    def __init__(self, hidden_size, args: argparse.Namespace):
        super(ExtraRelation, self).__init__()
        self.h_dense = nn.Linear(hidden_size * 2, hidden_size)
        self.t_dense = nn.Linear(hidden_size * 2, hidden_size)
        self.cls = GroupLinear(hidden_size, args.relation_num)
        self.loss_fn = CELoss()

    def forward(self, hidden_state, attention, ner_result, batch, is_test=False):
        batch_size = hidden_state.shape[0]
        attention = attention.mean(dim=1)
        head, tail = ner_result
        if is_test:
            entity_num = [len(x) for x in head]
        else:
            entity_num = batch['entity_mask'].sum(-1).tolist()
        pair_num = [x * (x - 1) for x in entity_num]

        all_h, all_t = [], []
        for i in range(batch_size):
            entity_feature = (hidden_state[i][head[i]] + hidden_state[i][tail[i]]) * 0.5
            entity_attention = (attention[i][head[i]] + attention[i][tail[i]]) * 0.5

            if entity_num[i] < 2:
                # all_h.append(torch.zeros((1, hidden_state.shape[-1] * 2), device=hidden_state.device))
                # all_t.append(torch.zeros((1, hidden_state.shape[-1] * 2), device=hidden_state.device))
                continue
            else:
                hts = torch.tensor(list(permutations(range(entity_num[i]), 2)),
                                   dtype=torch.int64, device=attention.device)
                head_feature, tail_feature = entity_feature[hts[:, 0]], entity_feature[hts[:, 1]]
                ht_attention = entity_attention[hts[:, 0]] * entity_attention[hts[:, 1]] * batch['attention_mask'][i]
                ht_attention = ht_attention / (torch.sum(ht_attention, dim=-1, keepdim=True) + 1e-20)
                ht_info = ht_attention @ hidden_state[i]

                all_h.append(torch.cat((head_feature, ht_info), dim=-1))
                all_t.append(torch.cat((tail_feature, ht_info), dim=-1))

        # all_h = torch.tanh(self.h_dense(pad_sequence(all_h, batch_first=True)))
        # all_t = torch.tanh(self.t_dense(pad_sequence(all_t, batch_first=True)))
        if is_test and not all_h:
            return None, [torch.tensor([]) for _ in range(batch_size)]
        all_h = torch.tanh(self.h_dense(torch.cat(all_h, dim=0)))
        all_t = torch.tanh(self.t_dense(torch.cat(all_t, dim=0)))
        pred = self.cls(all_h, all_t)

        if is_test:
            loss = None
        else:
            # mask = torch.zeros_like(batch['relations'], device=hidden_state.device, dtype=torch.bool)
            # for i in range(batch_size):
            #     mask[i, :pair_num[i]] = True
            mask = None
            loss = self.loss_fn(pred, batch['relations'], mask)
        pred = torch.argmax(pred, dim=-1)
        pred = torch.split(pred, pair_num, dim=0)
        return loss, pred


class ReModel(nn.Module):
    def __init__(self, args: argparse.Namespace):
        super(ReModel, self).__init__()
        self.bert = BertModel.from_pretrained(args.bert_path)
        self.bert.resize_token_embeddings(args.new_token_num)
        bert_hidden_size = self.bert.config.hidden_size
        self.ner_model = NER(bert_hidden_size, args)
        self.extra_relation = ExtraRelation(bert_hidden_size, args)

    def forward(self, batch, is_test=False):
        input_id = batch['input_id']
        attention_mask = batch['attention_mask']

        hidden_state, attention = process_long_input(self.bert, input_id, attention_mask, [101], [102])
        ner_loss, ner_result = self.ner_model(hidden_state, batch, is_test)
        re_loss, re_result = self.extra_relation(hidden_state, attention, ner_result, batch, is_test)

        if is_test:
            final_loss = None
        else:
            final_loss = ner_loss + re_loss

        return final_loss, ner_result, re_result
