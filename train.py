from argparse import ArgumentParser

from pytorch_lightning import Trainer

from src.module import NATODS


def main(args):
    trainer = Trainer.from_argparse_args(args)
    model = NATODS(hparams=args)
    trainer.fit(model)


if __name__ == '__main__':
    parser = ArgumentParser(add_help=False)
    parser = NATODS.add_model_specific_args(parser)
    parser = Trainer.add_argparse_args(parser)
    cli_args = parser.parse_args()
    main(cli_args)
