from argparse import Namespace, ArgumentParser
from typing import Dict, List
from pathlib import Path
from itertools import chain

import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from pytorch_lightning import LightningModule
from ipdb import set_trace

from src.preprocess import preprocess
from src.tokenizer import get_tokenizer, WordLevelTokenizer
from src.dataset import MultiWozDSTDataset
from src.positional_embedding import PositionalEncoding


def ctc_collapse(seq: List, padding) -> List:
    collapse = []
    prev = None
    for x in seq:
        if x == prev or x == padding:
            pass
        else:
            collapse.append(x)
        prev = x
    return collapse


class NATODS(LightningModule):
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser])

        # Embedding / Vocab
        parser.add_argument('--embedding_dim', default=256)

        parser.add_argument('--dropout', default=0.1)

        # Transformer Encoder
        parser.add_argument('--hidden_dim', default=256)
        parser.add_argument('--num_heads', default=4)
        parser.add_argument('--num_passes', default=2)

        # optimizer
        parser.add_argument('--lr', default=3e-4, type=float)
        parser.add_argument('--weight_decay', default=0, type=float)

        # data
        parser.add_argument('--input_multiplier', default=4, type=int,
                            help='# times to repeat input in order to make input longer than output')

        # dataset
        parser.add_argument('--train_path',
                            default='./.data/MultiWoz_2.1_NADST_Version/data2.1/nadst_train_dials.json')
        parser.add_argument('--val_path',
                            default='./.data/MultiWoz_2.1_NADST_Version/data2.1/nadst_dev_dials.json')
        parser.add_argument('--test_path',
                            default='./.data/MultiWoz_2.1_NADST_Version/data2.1/nadst_test_dials.json')
        parser.add_argument('--ontology_path',
                            default='./.data/MultiWoz_2.1_NADST_Version/data2.1/multi-woz/MULTIWOZ2.1/ontology.json')

        # tokenizer
        parser.add_argument('--tokenizer_path',
                            default='./tokenizer/multiwoz2.1-vocab.json')

        # dataloader
        parser.add_argument('--batch_size', default=128, type=int)

        return parser

    def __init__(self, hparams: Namespace):
        super().__init__()
        self.hparams = hparams

        self.criterion = nn.CTCLoss()

        if Path(self.hparams.tokenizer_path).exists():
            self.tokenizer = WordLevelTokenizer(self.hparams.tokenizer_path)
        else:
            train_turns = preprocess(self.hparams.train_path,
                                     self.hparams.ontology_path)
            self.tokenizer = get_tokenizer(train_turns,
                                           self.hparams.tokenizer_path)

        # embedding
        self.embedding = nn.Embedding(self.tokenizer.get_vocab_size(),
                                      self.hparams.embedding_dim)
        self.pos_embedding = PositionalEncoding(
            d_model=self.hparams.embedding_dim,
            dropout=self.hparams.dropout)

        # value decoder
        self.value_decoder = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=self.hparams.hidden_dim,
                num_heads=self.hparams.num_heads,
                dropout=self.hparams.dropout)
            for _ in range(3)])
        self.vocab_proj = nn.Linear(self.hparams.hidden_dim,
                                    self.tokenizer.get_vocab_size())

    def forward(self,
                batch: Dict[str, torch.Tensor]
                ) -> torch.Tensor:
        # history
        delex_history_embed = self.pos_embedding(
            self.embedding(batch['ids_delex_history']))
        history_embed = self.pos_embedding(
            self.embedding(batch['ids_history']))

        # value decoder's input
        fert_domain_embed = self.embedding(batch['ids_input_domain'])
        fert_slot_embed = self.embedding(batch['ids_input_slot'])
        fert_token_embed = self.pos_embedding(
            fert_domain_embed + fert_slot_embed)

        # value decooding
        z_fert_ds = self.forward_attentions(self.value_decoder,
                                            fert_token_embed,
                                            delex_history_embed,
                                            history_embed)
        value_logit = self.vocab_proj(z_fert_ds)
        return value_logit

    def forward_attentions(self, decoders, token_embed,
                           delex_history_embed, history_embed
                           ) -> torch.Tensor:
        assert len(decoders) == 3
        assert self.hparams.num_passes > 0
        z_ds_0 = token_embed
        for _ in range(self.hparams.num_passes):
            z_ds_1, _ = decoders[0].forward(
                query=z_ds_0, key=z_ds_0, value=z_ds_0,
            )
            z_ds_2, _ = decoders[1].forward(
                query=z_ds_1, key=delex_history_embed, value=delex_history_embed,
            )
            z_ds_3, _ = decoders[2].forward(
                query=z_ds_2, key=history_embed, value=history_embed,
            )
            z_ds_0 = z_ds_3 + z_ds_0
        return z_ds_0

    def calculate_loss(self,
                       logits: torch.Tensor,
                       batch: Dict
                       ) -> torch.Tensor:
        log_probs = logits.log_softmax(-1)
        intput_lengths = torch.LongTensor(
            [log_probs.size(0)] * log_probs.size(1))
        targets = batch['value'].T
        target_lengths = torch.sum(targets != self.train_dataset.ignore_idx, 1)
        assert all(intput_lengths[0] >= l for l in target_lengths), target_lengths
        loss = self.criterion(log_probs, targets,
                              intput_lengths, target_lengths)
        return loss

    def calculate_metrics(self,
                          outputs: List[Dict[ str, torch.Tensor]],
                          ) -> Dict[str, float]:
        metrics = {}
        pred_idxs = torch.cat([x['preds'] for x in outputs], 1).T
        preds_str = self.tokenizer.decode_batch(
            [ctc_collapse(seq, 0) for seq in pred_idxs.tolist()])
        value_str = list(chain.from_iterable([x['value_str'] for x in outputs]))
        assert len(value_str) == len(preds_str)
        metrics['accuracy/joint'] = \
            sum(1 if gt == pred else 0
                for gt, pred in zip(value_str, preds_str)) / len(value_str)
        return metrics

    def training_step(self, batch, batch_idx):
        logits = self.forward(batch)
        loss = self.calculate_loss(logits, batch)
        return {'loss': loss, 'log': {'loss/train': loss}}

    def validation_step(self, batch, batch_idx):
        logits = self.forward(batch)
        loss = self.calculate_loss(logits, batch)
        return {'loss': loss.cpu(),
                'preds': logits.cpu().argmax(-1),
                'value_str': batch['value_str'],
                }

    def validation_epoch_end(self, outputs, mode='val'):
        loss = torch.stack([x['loss'] for x in outputs]).mean()
        metrics = self.calculate_metrics(outputs)
        log = {
            **{f"loss/{mode}": loss},
            **{f'metrics/{mode}/{k}': v for k, v in metrics.items()}
        }
        return {f'{mode}_loss': loss, 'log': log}

    def test_step(self, *args, **kwargs):
        return self.validation_step(*args, **kwargs)

    def test_epoch_end(self, *args, **kwargs):
        return self.validation_epoch_end(*args, **kwargs, mode='test')

    def configure_optimizers(self):
        optimizer = Adam(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        return optimizer

    def prepare_data(self) -> None:
        train_turns = preprocess(self.hparams.train_path,
                                 self.hparams.ontology_path)
        val_turns = preprocess(self.hparams.val_path,
                               self.hparams.ontology_path)
        test_turns = preprocess(self.hparams.test_path,
                                self.hparams.ontology_path)
        self.train_dataset = MultiWozDSTDataset(train_turns,
                                                self.tokenizer,
                                                self.hparams.input_multiplier)
        self.val_dataset = MultiWozDSTDataset(val_turns,
                                              self.tokenizer,
                                              self.hparams.input_multiplier)
        self.test_dataset = MultiWozDSTDataset(test_turns,
                                               self.tokenizer,
                                               self.hparams.input_multiplier)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            collate_fn=self.train_dataset.collate_fn,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            collate_fn=self.val_dataset.collate_fn,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.hparams.batch_size,
            collate_fn=self.test_dataset.collate_fn,
        )
