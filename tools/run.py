import sys

import pytorch_lightning as pl
from helpers.pseudo_labels import generate_pseudo_labels
from pytorch_lightning.cli import LightningCLI


class MyLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.add_optimizer_args(
            nested_key="optimizer", link_to="model.init_args.optimizer_init")
        parser.add_lr_scheduler_args(
            nested_key="lr_scheduler", link_to="model.init_args.lr_scheduler_init")


def cli_main():

    if sys.argv[1] == "generate_pl":
        del sys.argv[1]
        sys.argv.append('--data.init_args.generate_pseudo_labels')
        sys.argv.append('True')
        cli = MyLightningCLI(pl.LightningModule,
                             pl.LightningDataModule,
                             subclass_mode_model=True,
                             subclass_mode_data=True,
                             save_config_kwargs={'overwrite': True},
                             seed_everything_default=2770466080,
                             run=False)
        generate_pseudo_labels(cli.model, cli.datamodule)
    
    else:
        cli = MyLightningCLI(pl.LightningModule,
                             pl.LightningDataModule,
                             subclass_mode_model=True,
                             subclass_mode_data=True,
                             save_config_kwargs={'overwrite': True},
                             seed_everything_default=2770466080)


if __name__ == '__main__':
    cli_main()
