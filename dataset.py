import os
import string
import random
import numpy as np
import math
import time
from tqdm import tqdm
import librosa

from torch.utils.data import DataLoader
import torch

def load_wav(wav_path, sr=22050):
    audio = librosa.core.load(wav_path, sr=sr)[0]
    return audio

class BaseDataset(torch.utils.data.Dataset):

    def __init__(self, hparams, fileid_list_path):
        self.hparams = hparams
        self.fileid_list = self.get_fileid_list(fileid_list_path)
        random.seed(hparams.seed)
        random.shuffle(self.fileid_list)

    def get_fileid_list(self, fileid_list_path):
        fileid_list = []
        with open(fileid_list_path, 'r') as f:
            for line in f.readlines():
                fileid_list.append(line.strip())

        return fileid_list

    def __len__(self):
        return len(self.fileid_list)

class VocoderDataset(BaseDataset):
    def __init__(self, hparams, feature_dirs, fileid_list_path):
        BaseDataset.__init__(self, hparams, fileid_list_path)
        self.feature_dirs = feature_dirs
        self.get_dirs(feature_dirs)

    def get_dirs(self, feature_dirs):
        self.mel_dir = feature_dirs[0]
        self.audio_dir = feature_dirs[1]

    def __getitem__(self, index):
        mel = np.load(os.path.join(self.mel_dir, self.fileid_list[index] + '.npy'))
        audio = load_wav(os.path.join(self.audio_dir, self.fileid_list[index] + '.wav'), self.hparams.sample_rate)
        return torch.FloatTensor(mel), torch.FloatTensor(audio)


class VocoderNoiseDataset(VocoderDataset):
    def __init__(self, hparams, feature_dirs, fileid_list_path):
        VocoderDataset.__init__(self, hparams, feature_dirs, fileid_list_path)
        self.noise_dir = feature_dirs[2]

    def __getitem__(self, index):
        mel = np.load(os.path.join(self.mel_dir, self.fileid_list[index] + '.npy'))
        audio = load_wav(os.path.join(self.audio_dir, self.fileid_list[index] + '.wav'), sr=self.hparams.sample_rate)
        noise = np.load(os.path.join(self.noise_dir, self.fileid_list[index] + '.npy'))

        return torch.FloatTensor(mel), torch.FloatTensor(audio), torch.FloatTensor(noise)

class VocoderCollate():

    def __init__(self, hparams):
        self.hparams = hparams
        self.mel_dim = self.hparams.mel_dim

    def __call__(self, batch):
        input_lengths, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([len(x[0]) for x in batch]),
            dim=0, descending=True)
        max_input_len = input_lengths[0]
        mel_padded = torch.FloatTensor(len(batch), self.mel_dim, self.hparams.segment_len)
        mel_padded.zero_()

        max_audio_len = self.hparams.segment_len * self.hparams.hop_size
        audio_padded = torch.FloatTensor(len(batch), 1, max_audio_len)
        audio_padded.zero_()

        for i in range(len(ids_sorted_decreasing)):
            mel = batch[ids_sorted_decreasing[i]][0]
            if mel.size(-1) == self.mel_dim:
                mel = mel.transpose(0, 1)
            audio = batch[ids_sorted_decreasing[i]][1].unsqueeze(0)

            if audio.size(-1) >= max_audio_len:
                mel_start = random.randint(0, mel.size(-1) - self.hparams.segment_len)
                mel = mel[:, mel_start:mel_start + self.hparams.segment_len]
                audio = audio[:, mel_start * self.hparams.hop_size: mel_start * self.hparams.hop_size + max_audio_len]
            else:
                mel = torch.nn.functional.pad(mel, (0, self.hparams.segment_len - mel.size(-1)), 'constant')
                audio = torch.nn.functional.pad(audio, (0, max_audio_len - audio.size(-1)), 'constant')

            mel_padded[i, :, :mel.size(-1)] = mel
            audio_padded[i, :, :audio.size(1)] = audio

        return mel_padded, audio_padded

class VocoderNoiseCollate():

    def __init__(self, hparams):
        self.hparams = hparams
        self.mel_dim = self.hparams.mel_dim

    def __call__(self, batch):
        input_lengths, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([len(x[0]) for x in batch]),
            dim=0, descending=True)

        max_input_len = input_lengths[0]
        mel_padded = torch.FloatTensor(len(batch), self.mel_dim, self.hparams.segment_len)
        mel_padded.zero_()

        max_audio_len = self.hparams.segment_len * self.hparams.hop_size
        audio_padded = torch.FloatTensor(len(batch), 1, max_audio_len)
        audio_padded.zero_()

        noise_padded = torch.FloatTensor(len(batch), 1, max_audio_len)
        noise_padded.zero_()

        output_lengths = torch.LongTensor(len(batch))

        for i in range(len(ids_sorted_decreasing)):
            mel = batch[ids_sorted_decreasing[i]][0]
            if mel.size(-1) == 80:
                mel = mel.transpose(0, 1)
            audio = batch[ids_sorted_decreasing[i]][1].unsqueeze(0)
            noise = batch[ids_sorted_decreasing[i]][2].unsqueeze(0)

            if audio.size(-1) >= max_audio_len:
                mel_start = random.randint(0, mel.size(-1) - self.hparams.segment_len)
                mel = mel[:, mel_start:mel_start + self.hparams.segment_len]
                audio = audio[:, mel_start * self.hparams.hop_size: mel_start * self.hparams.hop_size + max_audio_len]
                noise = noise[:, mel_start * self.hparams.hop_size: mel_start * self.hparams.hop_size + max_audio_len]
            else:
                mel = torch.nn.functional.pad(mel, (0, self.hparams.segment_len - mel.size(-1)), 'constant')
                audio = torch.nn.functional.pad(audio, (0, max_audio_len - audio.size(-1)), 'constant')
                noise = torch.nn.functional.pad(noise, (0, max_audio_len - noise.size(-1)), 'constant')

            mel_padded[i, :, :mel.size(-1)] = mel
            audio_padded[i, :, :audio.size(1)] = audio
            noise_padded[i, :, :noise.size(1)] = noise
            output_lengths[i] = mel.size(-1)

        return mel_padded, audio_padded, noise_padded

class DatasetConstructor():

    def __init__(self, hparams, num_replicas=1, rank=1):
        self.hparams = hparams
        self.num_replicas = num_replicas
        self.rank = rank

        self.dataset_function = {"VocoderDataset": VocoderDataset,
                                 "VocoderNoiseDataset": VocoderNoiseDataset}

        self.collate_function = {"VocoderCollate": VocoderCollate,
                                 "VocoderNoiseCollate": VocoderNoiseCollate}
        self._get_components()

    def _get_components(self):
        self._init_datasets()
        self._init_collate()
        self._init_data_loaders()

    def _init_datasets(self):
        self._train_dataset = self.dataset_function[self.hparams.dataset_type](self.hparams, self.hparams.feature_dirs, self.hparams.train_fileid_list_path)
        self._valid_dataset = self.dataset_function[self.hparams.dataset_type](self.hparams, self.hparams.feature_dirs, self.hparams.valid_fileid_list_path)

    def _init_collate(self):
        self._collate_fn = self.collate_function[self.hparams.collate_type](self.hparams)

    def _init_data_loaders(self):
        train_sampler = torch.utils.data.distributed.DistributedSampler(self._train_dataset, num_replicas=self.num_replicas, rank=self.rank, shuffle=True)

        self.train_loader = DataLoader(self._train_dataset, num_workers=8, shuffle=False,
                                       batch_size=self.hparams.batch_size, pin_memory=False,
                                       drop_last=True, collate_fn=self._collate_fn, sampler=train_sampler)

        self.valid_loader = DataLoader(self._valid_dataset, num_workers=8, shuffle=False,
                                       batch_size=self.hparams.batch_size, pin_memory=False,
                                       drop_last=True, collate_fn=self._collate_fn)

    def get_train_loader(self):
        return self.train_loader

    def get_valid_loader(self):
        return self.valid_loader
