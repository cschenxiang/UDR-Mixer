import subprocess
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from utils.dataset_utils import TrainDataset
from net.model import UDRMixer
from utils.schedulers import LinearWarmupCosineAnnealingLR
import numpy as np
#import wandb
from options import options as opt
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger,TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint


class Model(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.net = UDRMixer(dim=64, n_blocks=8, ffn_scale=2.0, upscaling_factor=2)
        self.loss_fn  = nn.L1Loss()
    
    def forward(self,x):
        return self.net(x)
    
    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        (degrad_patch, clean_patch) = batch
        restored = self.net(degrad_patch)

        loss = self.loss_fn(restored,clean_patch)
        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss)
        return loss
    
    def lr_scheduler_step(self,scheduler,*args, **kwargs):
        scheduler.step(self.current_epoch)
        lr = scheduler.get_lr()
    
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=2e-4)
        scheduler = LinearWarmupCosineAnnealingLR(optimizer=optimizer,warmup_epochs=15,max_epochs=150)

        return [optimizer],[scheduler]






def main():
    print("Options")
    print(opt)
    # if opt.wblogger is not None:
    #     logger  = WandbLogger(project=opt.wblogger,name="PromptIR-Train")
    # else:
    #     logger = TensorBoardLogger(save_dir = "logs/")

    trainset = TrainDataset(opt)
    checkpoint_callback = ModelCheckpoint(dirpath = opt.ckpt_dir,every_n_epochs = 1,save_top_k=-1)
    trainloader = DataLoader(trainset, batch_size=opt.batch_size, pin_memory=True, shuffle=True,
                             drop_last=True, num_workers=opt.num_workers)
    
    model = Model()
    
    trainer = pl.Trainer(max_epochs=opt.epochs,accelerator="gpu",devices=opt.num_gpus,callbacks=[checkpoint_callback])
    trainer.fit(model=model, train_dataloaders=trainloader)


if __name__ == '__main__':
    main()



