import os
import sys
import json
import argparse
import time
import math
import random
import numpy as np

import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.multiprocessing as mp

import utils
from dataset import DatasetConstructor
import model

class Trainer():

    def __init__(self, hparams):
        self.hparams = hparams
        self.init_random_seeds(hparams.seed)

        self.epoch = -1
        self.global_step = 0

    def init_random_seeds(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)       

    def train_one_epoch(self, rank, epoch, hparams, generator, optimizer_g, scheduler_g, data_loader, logger, writer):
        data_loader.sampler.set_epoch(epoch)
        
        for batch_idx, (mels, real_audio) in enumerate(data_loader):
            generator.train()
            start_t = time.perf_counter()
            mels = mels.cuda(rank, non_blocking=True)
            real_audio = real_audio.cuda(rank, non_blocking=True)
            noise = torch.randn(real_audio.shape).cuda(rank, non_blocking=True) * hparams.noise_scale

            predicted_score, target_score = generator(mels, real_audio, noise)

            optimizer_g.zero_grad()
           
            loss_score = torch.square(predicted_score - target_score).mean([1, 2]).mean()

            loss_g = loss_score
            loss_g.backward()
            optimizer_g.step()

            compute_time = time.perf_counter() - start_t
            if rank == 0:
                logger.info(f'Train Epoch: {epoch} [{batch_idx * self.hparams.batch_size}/{len(data_loader.dataset)} ({100. * batch_idx / len(data_loader):.0f}%)]\tLoss: {loss_g.item():.6f} time: {compute_time:.3f}s steps: {self.global_step}')

            self.global_step += 1

        if rank == 0 and epoch % hparams.writer_interval == 0:
            generator.eval()
            predicted_rk45_audio, _ = generator.inference(mels[:1], sampling_method='rk45')
            predicted_euler_1000_steps_audio, _ = generator.inference(mels[:1], sampling_steps=1000)
            predicted_euler_100_steps_audio, _ = generator.inference(mels[:1], sampling_steps=100)
            predicted_euler_10_steps_audio, _ = generator.inference(mels[:1], sampling_steps=10)
            generator.train()
            scalar_dict = {"loss/g/total": loss_g, "learning_rate": scheduler_g.get_last_lr()[0]}
            utils.summarize(
                writer=writer,
                global_step=self.global_step,
                audio={"p_rk45_audio": predicted_rk45_audio.cpu().numpy(),
                       "p_euler_1000_audio": predicted_euler_1000_steps_audio.cpu().numpy(),
                       "p_euler_100_audio": predicted_euler_100_steps_audio.cpu().numpy(),
                       "p_euler_10_audio": predicted_euler_10_steps_audio.cpu().numpy(),
                       "gt_audio": real_audio[0].cpu().numpy()
                },
                scalars=scalar_dict,
                hparams=hparams)

        scheduler_g.step()
        if rank == 0:
            logger.info('====> Epoch: {}'.format(epoch))
            
    def evaluate_one_epoch(self, rank, epoch, hparams, generator, data_loader, logger, writer):

        generator.eval()
        loss_score = 0.0
        losses_tot = []
        with torch.no_grad():
            for batch_idx, (mels, real_audio) in enumerate(data_loader):
                mels = mels.cuda(rank, non_blocking=True)
                real_audio = real_audio.cuda(rank, non_blocking=True)
                noise = torch.randn(real_audio.shape).cuda(rank, non_blocking=True) * hparams.noise_scale

                predicted_score, target_score = generator(mels, real_audio, noise)

                loss_score = torch.square(predicted_score - target_score).mean([1, 2]).mean()

                loss_gs = [loss_score]

                if batch_idx == 0:
                    losses_tot = loss_gs
                else:
                    losses_tot = [x + y for (x, y) in zip(losses_tot, loss_gs)]

                if rank == 0:
                    logger.info(f'Train Epoch: {epoch} [{batch_idx * self.hparams.batch_size}/{len(data_loader.dataset)} ({100. * batch_idx / len(data_loader):.0f}%)]\tLoss: {loss_score.item():.6f}')

        losses_tot = [x/len(data_loader) for x in losses_tot]
        loss_tot = sum(losses_tot)
        scalar_dict = {"loss/g/total": loss_tot}
        utils.summarize(
              writer=writer,
              global_step=self.global_step,
              scalars=scalar_dict)
        logger.info('====> Epoch: {}'.format(epoch))                
                

    def train(self, rank, hparams):

        if rank == 0:
            logger = utils.get_logger(hparams.model_dir)
            logger.info(hparams)
            writer = SummaryWriter(log_dir=os.path.join(hparams.model_dir, "train"))
            writer_eval = SummaryWriter(log_dir=os.path.join(hparams.model_dir, "eval"))

        torch.cuda.set_device(rank)

        dataset_constructor = DatasetConstructor(hparams, num_replicas=hparams.num_gpus, rank=rank)

        train_loader = dataset_constructor.get_train_loader()
        if rank == 0:
            valid_loader = dataset_constructor.get_valid_loader()

        generator = model.Generator(hparams).cuda(rank)
        g_parameters = list(generator.parameters())
        g_optimizer = optim.AdamW(g_parameters, lr=hparams.g_learning_rate, betas=(hparams.betas[0], hparams.betas[1]))

        checkpoint_path = utils.latest_checkpoint_path(hparams.model_dir, "M_*.pth")
        if os.path.isfile(checkpoint_path):
            self.epoch, self.global_step = utils.load_checkpoint(checkpoint_path, generator, g_optimizer)

        g_scheduler = torch.optim.lr_scheduler.ExponentialLR(g_optimizer, gamma=hparams.lr_decay, last_epoch=self.epoch)

        for epoch in range(self.epoch + 1, hparams.epochs):
            if rank==0:
                self.train_one_epoch(rank, epoch, hparams, generator, g_optimizer, g_scheduler, train_loader, logger, writer)
                self.evaluate_one_epoch(rank, epoch, hparams, generator, valid_loader, logger, writer_eval)
                if epoch % hparams.checkpoint_interval == 0:
                    utils.save_checkpoint(generator, g_optimizer, g_scheduler.get_lr(), epoch, self.global_step, os.path.join(hparams.model_dir, "M_{}.pth".format(epoch)))
            else:
                self.train_one_epoch(rank, epoch, hparams, generator, g_optimizer, g_scheduler, train_loader, None, None)        
                

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, help='Json file for configuration')
    parser.add_argument('-l', '--logdir', type=str, required=True)
    parser.add_argument('-m', '--model', type=str, required=True, help='Model name')

    args = parser.parse_args()
    hparams = utils.train_setup(args.config, args.logdir, args.model)

    trainer = Trainer(hparams)

    if hparams.num_gpus > 1:
        mp.spawn(trainer.train, nprocs=hparams.num_gpus, args=(hparams, ))
    else:
        trainer.train(0, hparams)


if __name__ == "__main__":
    main()
