import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence


class ATLoss(nn.Module):
    def __init__(self, args=None):
        super().__init__()

    @staticmethod
    def forward(pred, labels, label_mask=None):
        # if label_mask is not None:
        #     have_relation_num = label_mask.sum(-1)
        #     new_pred = [pred[i, :index] for i, index in enumerate(have_relation_num)]
        #     labels = [labels[i, :index] for i, index in enumerate(have_relation_num)]
        #     new_pred = torch.cat(new_pred, dim=0)
        #     labels = torch.cat(labels, dim=0)
        # else:
        #     new_pred = pred.reshape(-1, pred.shape[-1])
        #     labels = labels.reshape(-1, labels.shape[-1])
        tag_size = pred.shape[-1]
        if label_mask is not None:
            new_pred = pred[label_mask.unsqueeze(-1).repeat(1, 1, 1, tag_size)].reshape(-1, tag_size)
            labels = F.one_hot(labels[label_mask], tag_size)
        else:
            new_pred = pred.reshape(-1, pred.shape[-1])
            labels = F.one_hot(labels, tag_size)

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
    def __init__(self, args=None):
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


class FocalLoss(nn.Module):

    def __init__(self, args):
        super(FocalLoss, self).__init__()
        self.gamma = args.gamma
        self.num_class = args.tag_size

    def forward(self, pred, label, mask=None):
        # onehot_label = F.one_hot(label, num_classes=self.num_class)
        # pred = torch.masked_fill(logits, ~mask.unsqueeze(-1), -6e4)
        pred_softmax = F.softmax(pred, dim=-1)
        label = label.reshape(-1)

        pred_softmax = pred_softmax.reshape(-1, pred_softmax.shape[-1])
        pred_softmax = pred_softmax.gather(1, label.unsqueeze(-1))

        pred_log_softmax = torch.log(pred_softmax + 1e-5)

        p_loss = - torch.pow(1 - pred_softmax, self.gamma) * pred_log_softmax

        # n_pred_log_softmax = torch.log(1 - pred_softmax + 1e-5)
        # n_loss = - torch.pow(pred_softmax, self.gamma) * n_pred_log_softmax * (1 - onehot_label)

        loss = p_loss.squeeze(-1)  # + n_loss

        loss = loss * mask.reshape(-1)

        loss = loss.sum() / mask.sum()

        return loss


class MultiFocalLoss(nn.Module):
    """
    Focal_Loss= -1*alpha*((1-pt)**gamma)*log(pt)
    Args:
        num_class: number of classes
        alpha: class balance factor shape=[num_class, ]
        gamma: hyper-parameter
        reduction: reduction type
    """

    def __init__(self, args, reduction='mean'):
        super(MultiFocalLoss, self).__init__()
        self.gamma = args.gamma
        self.num_class = args.tag_size
        self.reduction = reduction
        self.smooth = 1e-4
        self.alpha = torch.ones(self.num_class) - 0.5

    def forward(self, logit, target, mask=None):
        logit = logit.reshape(-1, logit.shape[-1])
        alpha = self.alpha.to(logit.device)
        prob = F.softmax(logit, dim=-1)

        ori_shp = target.shape
        target = target.reshape(-1)

        prob = prob.gather(1, target).view(-1) + self.smooth  # avoid nan
        logpt = torch.log(prob)
        # alpha_class = alpha.gather(0, target.squeeze(-1))
        alpha_weight = alpha[target.squeeze().long()]
        loss = -alpha_weight * torch.pow(torch.sub(1.0, prob), self.gamma) * logpt

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'none':
            loss = loss.view(ori_shp)

        return loss


def multilabel_categorical_crossentropy(y_pred, y_true, mask=None):
    """多标签分类的交叉熵
    说明：y_true和y_pred的shape一致，y_true的元素非0即1，
         1表示对应的类为目标类，0表示对应的类为非目标类。
    警告：请保证y_pred的值域是全体实数，换言之一般情况下y_pred
         不用加激活函数，尤其是不能加sigmoid或者softmax！预测
         阶段则输出y_pred大于0的类。如有疑问，请仔细阅读并理解
         本文。
    """
    tag_size = y_pred.shape[-1]
    if mask is not None:
        y_pred = y_pred[mask.unsqueeze(-1).repeat(1, 1, 1, tag_size)].reshape(-1, tag_size)
        y_true = F.one_hot(y_true[mask], tag_size)

    y_pred = (1 - 2 * y_true) * y_pred
    y_pred_neg = y_pred - y_true * 1e12
    y_pred_pos = y_pred - (1 - y_true) * 1e12
    zeros = torch.zeros_like(y_pred[..., :1])
    y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1)
    y_pred_pos = torch.cat([y_pred_pos, zeros], dim=-1)
    neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
    pos_loss = torch.logsumexp(y_pred_pos, dim=-1)
    return (neg_loss + pos_loss).mean()
