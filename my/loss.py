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


class focal_loss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, num_classes=3, size_average=True):
        """
        focal_loss损失函数, -α(1-yi)**γ *ce_loss(xi,yi)
        步骤详细的实现了 focal_loss损失函数.
        :param alpha:   阿尔法α,类别权重.      当α是列表时,为各类别权重,当α为常数时,类别权重为[α, 1-α, 1-α, ....],常用于 目标检测算法中抑制背景类 , retainnet中设置为0.25
        :param gamma:   伽马γ,难易样本调节参数. retainnet中设置为2
        :param num_classes:     类别数量
        :param size_average:    损失计算方式,默认取均值
        """

        super(focal_loss, self).__init__()
        self.size_average = size_average
        if isinstance(alpha, list):
            assert len(alpha) == num_classes  # α可以以list方式输入,size:[num_classes] 用于对不同类别精细地赋予权重
            print("Focal_loss alpha = {}, 将对每一类权重进行精细化赋值".format(alpha))
            self.alpha = torch.Tensor(alpha)
        else:
            assert alpha < 1  # 如果α为一个常数,则降低第一类的影响,在目标检测中为第一类
            print(" --- Focal_loss alpha = {} ,将对背景类进行衰减,请在目标检测任务中使用 --- ".format(alpha))
            self.alpha = torch.zeros(num_classes)
            self.alpha[0] += alpha
            self.alpha[1:] += (1 - alpha)  # α 最终为 [ α, 1-α, 1-α, 1-α, 1-α, ...] size:[num_classes]
        self.gamma = gamma

    def forward(self, preds, labels, mask=None):
        """
        focal_loss损失计算
        :param preds:   预测类别. size:[B,N,C] or [B,C]    分别对应与检测与分类任务, B 批次, N检测框数, C类别数
        :param labels:  实际类别. size:[B,N] or [B]
        :return:
        """
        # assert preds.dim()==2 and labels.dim()==1
        preds = preds.view(-1, preds.size(-1))
        self.alpha = self.alpha.to(preds.device)
        preds_softmax = F.softmax(preds,
                                  dim=1)  # 这里并没有直接使用log_softmax, 因为后面会用到softmax的结果(当然你也可以使用log_softmax,然后进行exp操作)
        preds_logsoft = torch.log(preds_softmax)
        preds_softmax = preds_softmax.gather(1, labels.view(-1,
                                                            1))  # 这部分实现nll_loss ( crossempty = log_softmax + nll )
        preds_logsoft = preds_logsoft.gather(1, labels.view(-1, 1))
        self.alpha = self.alpha.gather(0, labels.view(-1))
        loss = -torch.mul(torch.pow((1 - preds_softmax), self.gamma),
                          preds_logsoft)  # torch.pow((1-preds_softmax), self.gamma) 为focal loss中 (1-pt)**γ
        loss = torch.mul(self.alpha, loss.t())
        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss
