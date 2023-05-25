import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from utils.cfg import py2cfg
import os
import torch
from src.lightning.lightning_fpn import FPNetModule
from src.lightning.lightning_segformer import SegFormerModule
from src.lightning.lightning_unetformer import UnetFormerModule
import numpy as np
import argparse
from pathlib import Path
from pytorch_lightning.loggers import WandbLogger
import random
import wandb

def seed(seed_num):
    random.seed(seed_num)
    os.environ['PYTHONHASHSEED'] = str(seed_num)
    np.random.seed(seed_num)
    torch.manual_seed(seed_num)
    torch.cuda.manual_seed(seed_num)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def get_args():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("-c", "--config_path", type=Path, help="Path to the config.")
    return parser.parse_args()

def main():
    args = get_args()
    config = py2cfg(args.config_path)
    seed(42)

    checkpoint_callback = ModelCheckpoint(save_top_k=config.save_top_k, monitor=config.monitor,
                                          save_last=config.save_last, mode=config.monitor_mode,
                                          dirpath=config.weights_path,
                                          filename=config.weights_name)

    logger = WandbLogger(project='UAVid_Semantic',name=config.experiment_name,log_model='all')

    if config.net_name == "Segformer":
        if config.pretrained_ckpt_path:
            model = SegFormerModule.load_from_checkpoint(config.pretrained_ckpt_path, config=config)
        else:
            model = SegFormerModule(config)
    elif config.net_name == "Unetformer":
        if config.pretrained_ckpt_path:
            model = UnetFormerModule.load_from_checkpoint(config.pretrained_ckpt_path, config=config)
        else:
            model = UnetFormerModule(config)
    elif config.net_name == "FPN":
        if config.pretrained_ckpt_path:
            model = FPNetModule.load_from_checkpoint(config.pretrained_ckpt_path, config=config)
        else:
            model = FPNetModule(config)


    trainer = pl.Trainer(devices=config.gpus, max_epochs=config.max_epoch, accelerator='auto',
                         check_val_every_n_epoch=config.check_val_every_n_epoch,
                         callbacks=[checkpoint_callback], strategy='auto',
                         enable_checkpointing=config.enable_checkpointing,precision=32,
                         logger=logger)


    trainer.fit(model=model)
    wandb.finish()


if __name__ == "__main__":
   main()
