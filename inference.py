import argparse
import random
import os
import time
from tqdm import tqdm
import numpy as np
import torch
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

    rtf_list = []
    for mel_name in tqdm(os.listdir(args.input)):
        mel_path = os.path.join(args.input, mel_name)
        wav_output_path = os.path.join(args.output, 'wavs', os.path.splitext(mel_name)[0] + '.wav')
        noise_output_path = os.path.join(args.output, 'noise', os.path.splitext(mel_name)[0] + '.npy')
        mel = np.load(mel_path)
        mel = torch.FloatTensor(mel).cuda().unsqueeze(0)
        if mel.shape[1] != hparams.mel_dim:
            mel = mel.transpose(1, 2)
        
        start_time = time.time()
        predicted_audio, noise = model.inference(mel, args.sampling_method, args.sampling_steps)
        predicted_audio, noise = predicted_audio.squeeze().cpu().numpy(), noise.squeeze().cpu().numpy()
        save_wav(predicted_audio, wav_output_path, hparams, norm=True)
        np.save(noise_output_path, noise)
                
        time_used = time.time() - start_time
        
        rtf = time_used / (predicted_audio.shape[0] / hparams.sample_rate)
        rtf_list.append(rtf)

    print(f'the average rtf is: {np.mean(rtf_list)}')                      
             
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hparams', type=str, required=True, help='Path to the checkpoints of models')
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

