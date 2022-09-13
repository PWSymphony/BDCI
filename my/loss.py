import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence


class ATLoss(nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(pred, labels, label_mask=None):
        if label_mask is not None:
            have_relation_num = label_mask.sum(-1)
            new_pred = [pred[i, :index] for i, index in enumerate(have_relation_num)]
            labels = [labels[i, :index] for i, index in enumerate(have_relation_num)]
            new_pred = torch.cat(new_pred, dim=0)
            labels = torch.cat(labels, dim=0)
        else:
            new_pred = pred.reshape(-1, pred.shape[-1])
            labels = labels.reshape(-1, labels.shape[-1])

        # TH label
        th_label = torch.zeros_like(labels, dtype=torch.float).to(labels)
        th_label[:, 0] = 1.0
        labels[:, 0] = 0.0

        p_mask = labels + th_label
        n_mask = 1 - labels

        # Rank positive classes to TH
        logit1 = new_pred - (1 - p_mask) * 1e30
        loss1 = -(F.log_softmax(logit1, dim=-1) * labels).sum(1)

        # Rank TH to negative classes
        logit2 = new_pred - (1 - n_mask) * 1e30
        loss2 = -(F.log_softmax(logit2, dim=-1) * th_label).sum(1)

        # Sum two parts
        loss = loss1 + loss2
        loss = loss.mean()
        return loss

    @staticmethod
    def get_label(logits, label_mask=None, num_labels=1):  # num_labels 是最大的标签数量
        L = None
        if label_mask is not None:
            have_relation_num = label_mask.sum(-1).cpu().detach().tolist()
            logits = [logits[i, :index] for i, index in enumerate(have_relation_num)]
            logits = torch.cat(logits, dim=0)
        else:
            have_relation_num = None
            L = logits.shape[1]
            logits = logits.reshape(-1, logits.shape[-1])

        th_logit = logits[:, 0].unsqueeze(1)
        output = torch.zeros_like(logits).to(logits)
        mask = (logits > th_logit)
        if num_labels > 0:
            top_v, _ = torch.topk(logits, num_labels, dim=1)
            top_v = top_v[:, -1]
            mask = (logits >= top_v.unsqueeze(1)) & mask
        output[mask] = 1.0
        output[:, 0] = (output.sum(1) == 0.).to(logits)

        if have_relation_num:
            output = torch.split(output, have_relation_num, dim=0)
            output = pad_sequence(output, batch_first=True)
        else:
            output = output.reshape(-1, L, output.shape[-1])
        return output


class CELoss(nn.Module):
    def __init__(self):
        super(CELoss, self).__init__()
        self.fn = nn.CrossEntropyLoss(reduction='none')

    def forward(self, pred, labels, mask=None):
        pred = pred.reshape(-1, pred.shape[-1])
        labels = labels.reshape(-1)
        loss = self.fn(pred, labels)
        if mask is not None:
            mask = mask.reshape(-1)
            loss = (loss * mask).sum() / mask.sum()
        else:
            loss = loss.mean()
        return loss

    @staticmethod
    def get_label(logits):
        return torch.max(logits, dim=-1)[1]
