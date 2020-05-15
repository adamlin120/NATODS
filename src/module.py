import logging
import pickle
from argparse import Namespace, ArgumentParser
from itertools import chain
from multiprocessing import cpu_count
from pathlib import Path
from typing import Dict

import torch
from convlab2.util.dataloader.dataset_dataloader import MultiWOZDataloader
from pytorch_lightning import LightningModule
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from src.data.multiwoz.dataset import MultiWozDataset
from src.data.vocab import TEXT_FIELD
from src.positional_embedding import PositionalEncoding


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
        parser.add_argument('--cached_dir', default='./cached/', type=Path)

        # dataloader
        parser.add_argument('--batch_size', default=64, type=int)

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

        self.text_field = TEXT_FIELD

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        # source: (source_seq_len x batch_size x embed_dim)
        domain = self.embedding(batch['domain']).transpose(0, 1)
        # slot: (source_seq_len x batch_size x embed_dim)
        slot = self.embedding(batch['slot']).transpose(0, 1)
        # embedded_source: (source_seq_len x batch_size x embed_dim)
        embedded_source = self.pos_embedding(domain + slot)
        # encoded_tokens: (source_seq_len x batch_size x d_model)
        encoded_tokens = self.encoder(embedded_source)
        # logits: (max_seq_len x batch_size x vocab_size)
        logits = self.proj(encoded_tokens)
        return logits

    def _calculate_loss(self,
                        logits: torch.Tensor,
                        batch: Dict[str, torch.Tensor]
                        ) -> torch.Tensor:
        # log_probs: (max_seq_len x batch_size x vocab_size)
        log_probs = logits.log_softmax(-1)
        # target: (batch_size x max_seq_len)
        target = batch['target'].transpose(0, 1)
        # source_lengths: (batch_size)
        source_lengths = batch['source_lengths']
        # target_lengths: (batch_size)
        target_lengths = batch['target_lengths']
        assert source_lengths.max() >= target_lengths.max()
        loss = self.criterion(
            log_probs=log_probs,
            targets=target,
            input_lengths=source_lengths,
            target_lengths=target_lengths,
        )
        return loss

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
        self.data = MultiWOZDataloader().load_data(
            data_dir="convlab_data/multiwoz/",
            data_key='all',
            role='sys',
            utterance=True,
            dialog_act=True,
            context=True,
            context_window_size=0,
            context_dialog_act=True,
            belief_state=True,
            last_opponent_utterance=True,
            last_self_utterance=True,
            ontology=True,
            session_id=True,
            span_info=True,
            terminated=True,
            goal=True)
        self.train_dataset = self._load_save_dataset('train')
        self.val_dataset = self._load_save_dataset('val')
        self.test_dataset = self._load_save_dataset('test')
        self._load_save_vocab()
        self._set_dataset_text_field()

    def _load_save_dataset(self, split: str) -> MultiWozDataset:
        cache_path: Path = self.hparams.cached_dir / f'{split}.pkl'
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        if cache_path.exists():
            logging.info(f"Loading {split} dataset cache at {cache_path}")
            dataset = pickle.load(cache_path.open('rb'))
        else:
            logging.info(f"Start creating {split} dataset...")
            dataset = MultiWozDataset(self.data[split])
            logging.info(f"Saving {split} dataset at {cache_path}")
            pickle.dump(dataset, cache_path.open('wb'))
        return dataset

    def _load_save_vocab(self):
        cache_path: Path = self.hparams.cached_dir / f'vocab.pkl'
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        if cache_path.exists():
            logging.info(f"Loading vocab at {cache_path}")
            vocab = pickle.load(cache_path.open('rb'))
            self.text_field.vocab = vocab
        else:
            logging.info(f"Creating vocab")
            source = list(chain.from_iterable(self.train_dataset.source))
            targets = [ins['target'].split(' ')
                       for ins in self.train_dataset + self.val_dataset]
            contexts = [ins['context'].split(' ')
                        for ins in self.train_dataset + self.val_dataset]
            self.text_field.build_vocab(source + targets + contexts,
                                        min_freq=1)
            pickle.dump(self.text_field.vocab, cache_path.open('wb'))
            logging.info(f"Saving vocab at {cache_path}")
        logging.info(f"Vocab size: {len(self.text_field.vocab)}")

    def _set_dataset_text_field(self):
        self.train_dataset.text_field = self.text_field
        self.val_dataset.text_field = self.text_field
        self.test_dataset.text_field = self.text_field

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            collate_fn=self.train_dataset.collate_fn,
            num_workers=cpu_count()
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            collate_fn=self.val_dataset.collate_fn,
            num_workers=cpu_count()
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.hparams.batch_size,
            collate_fn=self.test_dataset.collate_fn,
            num_workers=cpu_count()
        )
