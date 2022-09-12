import argparse
import logging
import platform
import warnings
from functools import partial

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from transformers import logging as transformer_log
from transformers.optimization import get_linear_schedule_with_warmup

from data import Data, get_batch, data_process, MyLogger  # , get_batch_single
from model import compute_score, ReModel

transformer_log.set_verbosity_error()
warnings.filterwarnings("ignore", category=UserWarning)


class PlModel(pl.LightningModule):
    def __init__(self, args: argparse.Namespace, total_step: int):
        super(PlModel, self).__init__()
        self.args = args
        self.total_step = total_step
        self.RE_model = ReModel(args)
        self.loss_list = []
        pl_logger = logging.getLogger('pytorch_lightning')
        pl_logger.addHandler(logging.FileHandler(args.save_name + '.txt'))
        self.save_hyperparameters(logger=True)

    def forward(self, batch):
        return self.RE_model(batch)

    def configure_optimizers(self):
        bert_params = [p for n, p in self.RE_model.named_parameters() if p.requires_grad and 'bert' in n]
        other_params = [p for n, p in self.RE_model.named_parameters() if p.requires_grad and 'bert' not in n]
        optimizer = torch.optim.AdamW([{"params": bert_params, "lr": self.args.bert_lr},
                                       {"params": other_params, "lr": self.args.lr}])
        scheduler = get_linear_schedule_with_warmup(optimizer=optimizer,
                                                    num_warmup_steps=int(self.total_step * self.args.warm_ratio),
                                                    num_training_steps=self.total_step)

        return {"optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler,
                                 "interval": 'step'}}

    def training_step(self, batch, batch_idx):
        ner_loss, ner_result, re_result = self.RE_model(batch)
        self.loss_list.append(float(ner_loss))

        cur_loss = round(sum(self.loss_list) / len(self.loss_list), 5)
        log_info = dict(loss=cur_loss)
        self.log_dict(log_info)

        return ner_loss

    def training_epoch_end(self, outputs):
        self.loss_list.clear()
        self.ner_f1.clear()

    def validation_step(self, batch, batch_idx):
        ner_loss, ner_result, re_result = self.RE_model(batch, is_test=True)
        f1_recall_precision = compute_score(ner_result, batch)
        return f1_recall_precision

    def validation_epoch_end(self, validation_step_outputs):
        f1 = round(sum(x[0] for x in validation_step_outputs) / len(validation_step_outputs), 5)
        recall = round(sum(x[1] for x in validation_step_outputs) / len(validation_step_outputs), 5)
        precision = round(sum(x[2] for x in validation_step_outputs) / len(validation_step_outputs), 5)
        log_info = dict(f1=f1, recall=recall, precision=precision)
        self.log_dict(log_info)


def main(args):
    # ========================================== 检查参数 ==========================================
    if not torch.cuda.is_available():
        args.accelerator = 'cpu'

    # ========================================== 获取数据 ==========================================
    data_process(args)
    data = Data(args.data_path)
    train_len = int(0.8 * len(data))
    train_data, val_data = data[:train_len], data[train_len:]

    num_workers = 4 if args.accelerator == 'gpu' and platform.system() == 'Linux' else 0
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, collate_fn=get_batch,
                              num_workers=num_workers)
    val_loader = DataLoader(val_data, batch_size=2, collate_fn=partial(get_batch, is_test=True),
                            num_workers=num_workers)

    # ========================================== 配置参数 ==========================================
    total_step = (len(train_loader) * args.max_epochs)
    strategy = None
    if args.accelerator == 'cpu':
        args.devices = None
        args.precision = 32
    elif args.accelerator == 'gpu' and len(args.devices) > 1:
        total_step //= len(args.devices)
        strategy = 'ddp'

    logger = MyLogger(args)
    model = PlModel(args, total_step)
    trainer = pl.Trainer.from_argparse_args(args=args, strategy=strategy, logger=logger, num_sanity_val_steps=0)

    # ========================================== 开始训练 ==========================================
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)


if __name__ == "__main__":
    """
    bert_name: [hfl/chinese-roberta-wwm-ext, nghuyong/ernie-3.0-base-zh]
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--accelerator", type=str, default='gpu')
    parser.add_argument("--accumulate_grad_batches", type=int, default=1)
    parser.add_argument("--devices", type=int, nargs='+', default=[0])
    parser.add_argument("--max_epochs", type=int, default=20)
    parser.add_argument("--precision", type=int, default=16)
    parser.add_argument("--log_every_n_steps", type=int, default=100)
    parser.add_argument("--enable_checkpointing", action='store_true')
    parser.add_argument("--enable_progress_bar", action='store_true')
    parser.add_argument("--gradient_clip_val", type=int, default=1)

    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--bert_lr", type=float, default=1e-5)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--warm_ratio", type=float, default=0.06)
    parser.add_argument("--bert_path", type=str, default="hfl/chinese-roberta-wwm-ext")
    parser.add_argument("--new_token_num", type=int, default=0)
    parser.add_argument("--save_name", type=str, default='test')
    parser.add_argument("--print_log", action='store_true')

    parser.add_argument("--max_len", type=int, default=1024)
    parser.add_argument("--tag_size", type=int, default=6)
    parser.add_argument("--relation_num", type=int, default=5)

    parser.add_argument("--raw_data_path", type=str, default='../BDCI_data/train.json')
    parser.add_argument("--data_path", type=str, default='data/raw.bin')

    train_args = parser.parse_args()
    main(train_args)
