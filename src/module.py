from argparse import Namespace, ArgumentParser
from typing import Dict

import ipdb
import torch
from pytorch_lightning import LightningModule
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from src.dataset import MultiWozDSTDataset
from src.positional_embedding import PositionalEncoding
from src.preprocess import preprocess
from src.tokenizer import get_tokenizer


class NATODS(LightningModule):
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser])

        # Embedding / Vocab
        parser.add_argument('--vocab_size', default=18905)
        parser.add_argument('--embedding_dim', default=256)

        # Transformer Encoder
        parser.add_argument('--num_layers', default=4)
        parser.add_argument('--d_model', default=256)
        parser.add_argument('--nhead', default=4)
        parser.add_argument('--dim_feedforward', default=512)

        # optimizer
        parser.add_argument('--lr', default=1e-5, type=float)
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
        parser.add_argument('--vocab_path', default='./vocab_file.json')

        # dataloader
        parser.add_argument('--batch_size', default=4, type=int)

        return parser

    def __init__(self, hparams: Namespace):
        super().__init__()
        self.hparams = hparams
        self.criterion = nn.CTCLoss()
        self.embedding = nn.Embedding(self.hparams.vocab_size,
                                      self.hparams.embedding_dim)
        self.pos_embedding = PositionalEncoding(
            d_model=self.hparams.embedding_dim)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.hparams.d_model,
                nhead=self.hparams.nhead,
                dim_feedforward=self.hparams.dim_feedforward),
            num_layers=self.hparams.num_layers)
        self.proj = nn.Linear(self.hparams.d_model, self.hparams.vocab_size)

    def forward(self,
                batch: Dict[str, torch.Tensor]
                ) -> torch.Tensor:
        ipdb.set_trace()
        pass

    def _calculate_loss(self,
                        logits: torch.Tensor,
                        batch: Dict
                        ) -> torch.Tensor:
        pass

    def training_step(self, batch, batch_idx):
        logits = self.forward(batch)
        loss = self._calculate_loss(logits, batch)
        return {'loss': loss, 'log': {'train/loss': loss}}

    def validation_step(self, batch, batch_idx):
        logits = self.forward(batch)
        loss = self._calculate_loss(logits, batch)
        return {'val_loss': loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val/loss': avg_loss}
        return {'val_loss': avg_loss, 'log': tensorboard_logs}

    def test_step(self, batch, batch_idx):
        logits = self.forward(batch)
        return {'logits': logits}

    def test_epoch_end(self, outputs):
        return {}

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
        self.tokenizer = get_tokenizer(train_turns, self.hparams.vocab_path)
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
