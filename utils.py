import os
import glob
import sys
import base64
import hashlib
import requests
import itertools
import argparse
import logging
import json
import yaml
import subprocess
import numpy as np
from scipy.io.wavfile import read
import torch

MATPLOTLIB_FLAG = False

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging

numba_logger = logging.getLogger('numba')
numba_logger.setLevel(logging.WARNING)



def load_checkpoint(checkpoint_path, model, optimizer=None):
    assert os.path.isfile(checkpoint_path)
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    epoch = 1
    if 'epoch' in checkpoint_dict.keys():
        epoch = checkpoint_dict['epoch']
    if 'learning_rate' in checkpoint_dict.keys():
        learning_rate = checkpoint_dict['learning_rate']
    if 'global_step' in checkpoint_dict.keys():
        global_step = checkpoint_dict['global_step']
    else:
        global_step = 0
    if optimizer is not None and 'optimizer' in checkpoint_dict.keys():
        optimizer.load_state_dict(checkpoint_dict['optimizer'])

    saved_state_dict = checkpoint_dict['generator']
    state_dict = model.state_dict()
    new_state_dict = {}
    for k, v in state_dict.items():
        try:
            new_state_dict[k] = saved_state_dict[k]
        except:
            logger.info("%s is not in the checkpoint" % k)
            new_state_dict[k] = v
    model.load_state_dict(new_state_dict)
    logger.info("Loaded checkpoint '{}' (epoch {})" .format(
        checkpoint_path, epoch))
    return epoch, global_step


def latest_checkpoint_path(dir_path, regex="M_*.pth"):
    f_list = glob.glob(os.path.join(dir_path, regex))
    f_list.sort(key=lambda f: int("".join(filter(str.isdigit, f))))
    if len(f_list) == 0:
        return ""
    x = f_list[-1]
    return x


def get_hparams_from_dir(model_dir):
    config_save_path = os.path.join(model_dir, "config.yaml")
    with open(config_save_path, "r") as f:
        data = f.read()
    config = yaml.full_load(data)

    hparams = HParams(**config)
    hparams.model_dir = model_dir
    return hparams


def save_checkpoint(generator, optimizer, learning_rate, epoch, global_step, checkpoint_path):
    logger.info("Saving model and optimizer state at epoch {} to {}".format(
        epoch, checkpoint_path))
    torch.save({'generator': generator.state_dict(),
                'epoch': epoch,
                'optimizer': optimizer.state_dict(),
                'learning_rate': learning_rate,
                'global_step': global_step}, checkpoint_path)


def summarize(writer, global_step, scalars={}, histograms={}, images={}, audio={}, hparams=None):
    for k, v in scalars.items():
        writer.add_scalar(k, v, global_step)
    for k, v in histograms.items():
        writer.add_histogram(k, v, global_step)
    for k, v in images.items():
        writer.add_image(k, v, global_step, dataformats='HWC')
    for k, v in audio.items():
        writer.add_audio(k, v, global_step, hparams.sample_rate)


def plot_spectrogram_to_numpy(spectrogram):
    global MATPLOTLIB_FLAG
    if not MATPLOTLIB_FLAG:
        import matplotlib
        matplotlib.use("Agg")
        MATPLOTLIB_FLAG = True
        mpl_logger = logging.getLogger('matplotlib')
        mpl_logger.setLevel(logging.WARNING)
    import matplotlib.pylab as plt
    import numpy as np

    fig, ax = plt.subplots()
    im = ax.imshow(spectrogram, aspect="auto", origin="lower",
                   interpolation='none')
    plt.colorbar(im, ax=ax)
    plt.xlabel("Frames")
    plt.ylabel("Channels")
    plt.tight_layout()

    fig.canvas.draw()
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close()
    return data


def get_logger(model_dir, filename="train.log"):
    global logger
    logger = logging.getLogger(os.path.basename(model_dir))
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter("%(asctime)s\t%(name)s\t%(levelname)s\t%(message)s")
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    h = logging.FileHandler(os.path.join(model_dir, filename))
    h.setLevel(logging.DEBUG)
    h.setFormatter(formatter)
    logger.addHandler(h)
    return logger


def train_setup(config, logdir, model):

    model_dir = os.path.join(logdir, model)

    if not os.path.exists(model_dir):
        os.makedirs(model_dir, exist_ok=True)

    config_save_path = os.path.join(model_dir, "config.yaml")
    with open(config, "r") as f:
        data = f.read()
    with open(config_save_path, "w") as f:
        f.write(data)

    config = yaml.full_load(data)

    hparams = HParams(**config)
    hparams.model_dir = model_dir
    return hparams


def config_setup(config):
    with open(config, "r") as f:
        data = f.read()

    config = yaml.full_load(data)

    hparams = HParams(**config)
    return hparams


class HParams():
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if type(v) == dict:
                v = HParams(**v)
            self[k] = v

    def get(self, key, default):
        if self.__contains__(key):
            return self.__getitem__(key)
        else:
            return default

    def keys(self):
        return self.__dict__.keys()

    def items(self):
        return self.__dict__.items()

    def values(self):
        return self.__dict__.values()

    def __len__(self):
        return len(self.__dict__)

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        return setattr(self, key, value)

    def __contains__(self, key):
        return key in self.__dict__

    def __repr__(self):
        return self.__dict__.__repr__()

