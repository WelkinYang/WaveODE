import argparse
import random
import os
import time
from tqdm import tqdm
import numpy as np
import torch
import uuid
from scipy.io import wavfile

from model import Generator
from utils import config_setup, load_checkpoint

def save_wav(wav, path, hparams, norm=False):
    if norm:
        wav *= 32767 / max(0.01, np.max(np.abs(wav)))
        wavfile.write(path, hparams.sample_rate, wav.astype(np.int16))
    else:
        sf.write(path, wav, hparams.sample_rate)

def Synthesize(args, hparams):

    model = Generator(hparams).cuda()
    load_checkpoint(args.checkpoint, model)
    model.eval()

    os.makedirs(args.output, exist_ok=True)
    os.makedirs(os.path.join(args.output, 'wavs'), exist_ok=True)
    os.makedirs(os.path.join(args.output, 'noise'), exist_ok=True)
    os.makedirs(os.path.join(args.output, 'mels'), exist_ok=True)

    rtf_list = []
    batch_queue = []
    input_lengths = []
    for mel_name in tqdm(os.listdir(args.input)):
        mel_path = os.path.join(args.input, mel_name)
        mel = np.load(mel_path)
        mel = torch.from_numpy(mel)
        batch_queue.append(mel)
        input_lengths.append(mel.shape[0])

        if len(batch_queue) == args.batch_size:
            mels = torch.zeros((args.batch_size, hparams.mel_dim, args.segment_len), dtype=torch.float32).cuda()
            for i, mel in enumerate(batch_queue):
                mel = mel.transpose(1, 0)
                if mel.shape[1] > args.segment_len:
                    mel_start = random.randint(0, mel.shape[1] - args.segment_len)
                    mel = mel[:, mel_start:mel_start + args.segment_len]
                else:
                    mel = torch.nn.functional.pad(mel, (0, args.segment_len - mel.shape[1]), 'constant') 

                mels[i] = mel

             
            predicted_audios, noises = model.inference(mels, args.sampling_method, args.sampling_steps)
            predicted_audios = predicted_audios.squeeze().cpu().numpy()
            noises = noises.squeeze().cpu().numpy()
            mels = mels.squeeze().cpu().numpy()

            for predicted_audio, noise, mel in zip(predicted_audios, noises, mels):
                fileid = str(uuid.uuid1())

                wav_output_path = os.path.join(args.output, 'wavs',  fileid + '.wav')
                noise_output_path = os.path.join(args.output, 'noise', fileid + '.npy')                
                mel_output_path = os.path.join(args.output, 'mels', fileid + '.npy')

                predicted_audio = predicted_audio[:input_lengths[i] * hparams.hop_size]
                noise = noise[:input_lengths[i] * hparams.hop_size]
                mel = mel[:, :input_lengths[i]]
 
                save_wav(predicted_audio, wav_output_path, hparams, norm=True)
                np.save(noise_output_path, noise)
                np.save(mel_output_path, mel)

            batch_queue = []
            input_lengths = []                                
             
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hparams', type=str, required=True, help='Path to the checkpoints of models')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size of inference')
    parser.add_argument('--segment_len', type=int, default=128, help='Segment length of batch inference')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--input', type=str, required=True, help='Path to input folder')
    parser.add_argument('--output', type=str, required=True, help='Path to output folder')
    parser.add_argument('--sampling_method', type=str, default='rk45')
    parser.add_argument('--sampling_steps', type=int, default=1000)
   
    args = parser.parse_args()

    hparams = config_setup(args.hparams)
    Synthesize(args, hparams)

if __name__ == "__main__":
    main() 

