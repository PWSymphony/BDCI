import argparse
import json
import logging
import platform
import warnings

import pytorch_lightning as pl
import torch
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
from transformers import logging as transformer_log
from transformers.optimization import get_linear_schedule_with_warmup

from data import Data, get_batch, data_process, MyLogger, get_result
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
        self.save_name = args.save_name

    def forward(self, batch):
        return self.RE_model(batch)

    def configure_optimizers(self):
        bert_params = [p for n, p in self.RE_model.named_parameters() if p.requires_grad and 'bert' in n]
        other_params = [p for n, p in self.RE_model.named_parameters() if p.requires_grad and 'bert' not in n]
        optimizer = torch.optim.AdamW([{"params": bert_params, "lr": self.args.bert_lr, "weight_decay": 1e-4},
                                       {"params": other_params, "lr": self.args.lr,
                                        "weight_decay": self.args.weight_decay}])
        scheduler = get_linear_schedule_with_warmup(optimizer=optimizer,
                                                    num_warmup_steps=int(self.total_step * self.args.warm_ratio),
                                                    num_training_steps=self.total_step)

        return {"optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler,
                                 "interval": 'step'}}

    def training_step(self, batch, batch_idx):
        ner_loss, _, _ = self.RE_model(batch)
        self.loss_list.append(float(ner_loss))

        cur_loss = round(sum(self.loss_list) / len(self.loss_list), 5)
        log_info = dict(loss=cur_loss)
        self.log_dict(log_info)

        return ner_loss

    def training_epoch_end(self, outputs):
        self.loss_list.clear()

    def validation_step(self, batch, batch_idx):
        ner_loss, ner_result, re_result = self.RE_model(batch, is_test=True)
        true_recall_pred = compute_score(batch, ner_result, re_result)
        return true_recall_pred

    def validation_epoch_end(self, validation_step_outputs):
        ner_true = sum(x[0] for x in validation_step_outputs)
        ner_recall = sum(x[1] for x in validation_step_outputs)
        ner_pred = sum(x[2] for x in validation_step_outputs)
        re_true = sum(x[3] for x in validation_step_outputs)
        re_recall = sum(x[4] for x in validation_step_outputs)
        re_pred = sum(x[5] for x in validation_step_outputs)

        ner_recall = ner_true / (ner_recall + 1e-20)
        ner_precision = ner_true / (ner_pred + 1e-20)
        ner_f1 = 2 * ner_recall * ner_precision / (ner_recall + ner_precision + 1e-20)

        re_recall = re_true / (re_recall + 1e-20)
        re_precision = re_true / (re_pred + 1e-20)
        re_f1 = 2 * re_recall * re_precision / (re_recall + re_precision + 1e-20)

        log_info = dict(ner_f1=ner_f1, ner_recall=ner_recall, ner_precision=ner_precision,
                        re_f1=re_f1, re_recall=re_recall, re_precision=re_precision)
        self.log_dict(log_info)

    def test_step(self, batch, batch_idx):
        ner_loss, ner_result, re_result = self.RE_model(batch, is_test=True)
        return get_result(ner_result, re_result, batch)

    def test_epoch_end(self, outputs):
        res = [x for batch in outputs for x in batch]
        with open(self.save_name + '_result.json', 'w') as f:
            for r in res:
                f.write(json.dumps(r, ensure_ascii=False))
                f.write('\n')


def main(args):
    # ========================================== 检查参数 ==========================================
    seed_everything(args.seed)
    if not torch.cuda.is_available():
        args.accelerator = 'cpu'

    # ========================================== 获取数据 ==========================================
    train_data, val_data, test_data = data_process(args)
    train_data = Data(train_data)
    val_data = Data(val_data)
    test_data = Data(test_data)

    num_workers = 4 if args.accelerator == 'gpu' and platform.system() == 'Linux' else 0
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, collate_fn=get_batch,
                              num_workers=num_workers)
    val_loader = DataLoader(val_data, batch_size=6, collate_fn=get_batch, num_workers=num_workers)
    test_loader = DataLoader(test_data, batch_size=6, collate_fn=get_batch, num_workers=num_workers)

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

    callbacks = []
    if args.enable_checkpointing:
        checkpoint_callback = ModelCheckpoint(save_top_k=2,
                                              monitor="re_f1",
                                              mode="max",
                                              dirpath='checkpoint',
                                              filename=args.save_name + '--{epoch}--{re_f1:.4f}',
                                              save_weights_only=True,
                                              auto_insert_metric_name=True)
        callbacks.append(checkpoint_callback)

    trainer = pl.Trainer.from_argparse_args(args=args, strategy=strategy, logger=logger,
                                            num_sanity_val_steps=0, callbacks=callbacks)

    # ========================================== 开始训练 ==========================================
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    # model = model.load_from_checkpoint(r'checkpoint/Roberta_base--epoch=30--re_f1=0.6637.ckpt')
    # trainer.test(model=model, dataloaders=test_loader)


if __name__ == "__main__":
    """
    bert_name: [hfl/chinese-roberta-wwm-ext, nghuyong/ernie-3.0-base-zh]
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=6)
    parser.add_argument("--accelerator", type=str, default='gpu')
    parser.add_argument("--accumulate_grad_batches", type=int, default=1)
    parser.add_argument("--devices", type=int, nargs='+', default=[6, 7])
    parser.add_argument("--max_epochs", type=int, default=30)
    parser.add_argument("--precision", type=int, default=16)
    parser.add_argument("--log_every_n_steps", type=int, default=100)
    parser.add_argument("--enable_checkpointing", action='store_true')
    parser.add_argument("--enable_progress_bar", action='store_true')
    parser.add_argument("--gradient_clip_val", type=int, default=1)

    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--bert_lr", type=float, default=5e-5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warm_ratio", type=float, default=0.06)
    parser.add_argument("--bert_path", type=str, default="hfl/chinese-roberta-wwm-ext")
    parser.add_argument("--new_token_num", type=int, default=0)
    parser.add_argument("--save_name", type=str, default='test')
    parser.add_argument("--print_log", action='store_true')

    parser.add_argument("--max_len", type=int, default=1024)
    parser.add_argument("--dis_emb", type=int, default=100)
    parser.add_argument("--type_emb", type=int, default=100)
    parser.add_argument("--tag_size", type=int, default=5)
    parser.add_argument("--max_entity", type=int, default=100)
    parser.add_argument("--max_entity_len", type=int, default=20)
    parser.add_argument("--relation_num", type=int, default=5)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--gamma", type=float, default=2)

    parser.add_argument("--ratio", type=float, default=0.8)
    parser.add_argument("--raw_data_path", type=str, default='../BDCI_data/train.json')
    parser.add_argument("--raw_test_data", type=str, default='../BDCI_data/evalA.json')

    train_args = parser.parse_args()
    main(train_args)