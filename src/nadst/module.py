from argparse import Namespace, ArgumentParser
from typing import Dict, List, Tuple
from pathlib import Path

import torch
from torch import nn
from torch.nn.functional import cross_entropy
from torch.optim import Adam
from torch.utils.data import DataLoader
from pytorch_lightning import LightningModule
from sklearn.metrics import accuracy_score, f1_score

from src.preprocess import preprocess
from src.tokenizer import get_tokenizer, WordLevelTokenizer
from src.dataset import MultiWozDSTDataset
from src.positional_embedding import PositionalEncoding


class NADST(LightningModule):
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

        # output layer
        parser.add_argument('--num_gate', default=3)
        parser.add_argument('--num_fertility', default=9)

        # optimizer
        parser.add_argument('--lr', default=3e-4, type=float)
        parser.add_argument('--weight_decay', default=0, type=float)

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

        # gate fertility decoder
        self.gate_fert_decoders = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=self.hparams.hidden_dim,
                num_heads=self.hparams.num_heads,
                dropout=self.hparams.dropout)
            for _ in range(3)])
        self.gate_proj = nn.Linear(self.hparams.hidden_dim,
                                   self.hparams.num_gate)
        self.fertility_proj = nn.Linear(self.hparams.hidden_dim,
                                        self.hparams.num_fertility)
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
                ) -> Dict[str, torch.Tensor]:
        # history
        delex_history_embed = self.pos_embedding(
            self.embedding(batch['ids_delex_history']))
        history_embed = self.pos_embedding(
            self.embedding(batch['ids_history']))

        # gate, fertility decoder's input
        domain_embed = self.embedding(batch['ids_input_domain'])
        slot_embed = self.embedding(batch['ids_input_slot'])
        token_embed = domain_embed + slot_embed

        # gate, fertility decoding
        z_ds = self.forward_attentions(self.gate_fert_decoders,
                                       token_embed,
                                       delex_history_embed,
                                       history_embed)
        fertility_logit = self.fertility_proj(z_ds)
        gate_logit = self.gate_proj(z_ds)

        # value decoder's input
        fert_domain_embed = self.embedding(batch['ids_input-fert_domain'])
        fert_slot_embed = self.embedding(batch['ids_input-fert_slot'])
        fert_token_embed = self.pos_embedding(
            fert_domain_embed + fert_slot_embed)

        # value decooding
        z_fert_ds = self.forward_attentions(self.value_decoder,
                                            fert_token_embed,
                                            delex_history_embed,
                                            history_embed)
        value_logit = self.vocab_proj(z_fert_ds)

        return {
            'fertility': fertility_logit,
            'gate': gate_logit,
            'value': value_logit
        }

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
                       logits: Dict[str, torch.Tensor],
                       batch: Dict
                       ) -> Dict[str, torch.Tensor]:
        loss = {
            pred: cross_entropy(
                logit.view(-1, logit.size(-1)),
                batch[pred].reshape(-1))
            for pred, logit in logits.items()
        }
        loss['sum'] = loss['gate'] + loss['fertility'] + loss['value']
        return loss

    def calculate_metrics(self,
                          preds: Dict[str, torch.Tensor],
                          labels: Dict,
                          values: Dict[str, List[Tuple[str, str]]]
                          ) -> Dict[str, float]:
        metrics = {}
        # gate, fertility
        for pred in ['gate', 'fertility']:
            metrics[f'accuracy/{pred}'] = accuracy_score(
                labels[pred].view(-1), preds[pred].view(-1))
            metrics[f'f1/{pred}'] = f1_score(
                labels[pred].view(-1), preds[pred].view(-1), average='macro')
        # values
        for d_s, vs in values.items():
            metrics[f'accuracy/{d_s}'] = \
                sum(1 for pred, gt in vs if pred == gt) / len(vs) if vs else 0
        return metrics

    def training_step(self, batch, batch_idx):
        logits = self.forward(batch)
        losses = self.calculate_loss(logits, batch)
        log = {f"loss/train/{k}": v for k, v in losses.items()}
        return {'loss': losses['sum'], 'log': log}

    def validation_step(self, batch, batch_idx):
        logits = self.forward(batch)
        loss = self.calculate_loss(logits, batch)
        logits = {k: v.cpu() for k, v in logits.items()}

        values = {k: [] for k in self.val_dataset.slot_names + ['joint']}
        for value_logit, gt_encoded_value, fertility, domain, slot in zip(
                logits['value'].transpose(0, 1),
                batch[f'encoded_value'],
                batch['fertility'].sum(0),
                batch['input-fert_domain'],
                batch['input-fert_slot']
        ):
            domain: List[str] = domain.split()
            slot: List[str] = slot.split()
            pred_values: List[str] = self.tokenizer.decode(
                value_logit.argmax(-1).tolist(),
                skip_special_tokens=False).split(' ')[:fertility]
            gt_values: List[str] = gt_encoded_value.tokens[:fertility]
            assert fertility == len(domain) == len(slot) == \
                   len(pred_values) == len(gt_values)
            for d, s, pred, gt in zip(domain, slot, pred_values, gt_values):
                values[f'{d}_{s}'].append((pred, gt))
            values['joint'].append((pred_values, gt_values))
        return {'loss': loss,
                'logits': logits,
                'batch': batch,
                'values': values}

    def validation_epoch_end(self, outputs, mode='val'):
        # gate, fertility
        pred = {key: torch.cat([x['logits'][key] for x in outputs],
                               dim=1).argmax(-1)
                for key in ['gate', 'fertility']}
        y_true = {key: torch.cat([x['batch'][key] for x in outputs],
                                 dim=-1).cpu()
                  for key in ['gate', 'fertility']}

        # values
        values = {k: [] for k in self.val_dataset.slot_names + ['joint']}
        for x in outputs:
            for k, v in x['values'].items():
                values[k].extend(v)

        metrics = self.calculate_metrics(pred, y_true, values)
        loss = {
            f'{mode}/{key}': torch.stack([x['loss'][key] for x in outputs]).mean()
            for key in outputs[0]['loss']
        }
        log = {
            **{f"loss/{k}": v for k, v in loss.items()},
            **{f'metrics/{mode}/{k}': v for k, v in metrics.items()}
        }
        return {f'{mode}_loss': loss[f'{mode}/sum'], 'log': log}

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
        self.train_dataset = MultiWozDSTDataset(train_turns, self.tokenizer)
        self.val_dataset = MultiWozDSTDataset(val_turns, self.tokenizer)
        self.test_dataset = MultiWozDSTDataset(test_turns, self.tokenizer)

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
