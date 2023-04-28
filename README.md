# WaveODE
An ODE-based generative neural vocoder using Rectified Flow

## Introduction
Recently ODE-based generative models are a hot topic in machine learning and image generation and have achieved remarkable performance. However, due to the differences in data distribution between images and waveforms, it is not clear how well these models perform on speech tasks. In this project, I implement an ODE-based generative neural coder called WaveODE using Rectified Flow [4] as the backbone and hope to contribute to the generalization of ODE-based generative models for speech tasks.

## Pre-requisites
* The testdata folder contains some example files that allow the project to run directly.
* If you want to run with your own dataset:
  1. Replace the feature_dirs and fileid_list in config.json with your own dataset.
  2. Modify the acoustic parameters to match the data you are using and adjust the batch size to the number you need.

## Training and inference

### Train WaveODE with 1-Rectified Flow from scratch
```
python3 -u train.py -c config.yaml -l logdir -m waveode_1-rectified_flow
```

### Inference
1. RK45 solver: 
```
python3 inference.py --hparams config.yaml --checkpoint logdir/waveode_1-rectified_flow/xxx.pth --input test_mels_dir  --output out_dir
```
2. Euler sover: 
```
python3 inference.py --hparams config.yaml --checkpoint logdir/waveode_1-rectified_flow/xxx.pth --input test_mels_dir  --output out_dir --sampling_method euler --sampling_steps N
```

### Train WaveODE with 2-Rectified Flow
1. Generate (noise, audio) tuples using 1-Rectified Flow: 
```
python3 inferene.py --hparams config.yaml --checkpoint logdir/waveode/xxx.pth --input all_mels_dir  --output testdata/generate
```
2. Train 2-Rectified Flow using generated data
```
python3 -u train_reflow.py -c config_reflow.yaml -l logdir -m waveode_2-rectified_flow
```

## Q&A

### What is ODE-based generative models?

ODE-based generative model (also known as continuous normalizing flow) is a family of generative models that use an ODE-based model to model data distributions where the trajectory from an initial distribution such as a Gaussian distribution to a target distribution follows a ordinary differential equation. 

There are some relevant papers:

[1] Neural ordinary differential equations (Chen et al. 2018) [Paper](https://arxiv.org/abs/1806.07366)

[2] FFJORD: Free-Form Continuous Dynamics for Scalable Reversible Generative Models (Grathwohl et al. 2018) [Paper](https://arxiv.org/abs/1810.01367)

[3] Score-Based Generative Modeling through Stochastic Differential Equations (Song et al. 2021) [Paper](https://arxiv.org/abs/2011.13456)

[3] Flow Matching for Generative Modeling (Lipman et al. 2023) [Paper](https://openreview.net/forum?id=PqvMRDCJT9t)

[4] Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow (Liu et al. 2023) [Paper](https://openreview.net/forum?id=XVjTT1nw5z)

[5] Stochastic Interpolants: A Unifying Framework for Flows and Diffusions (Albergo et al. 2023) [Paper](https://arxiv.org/abs/2303.08797)

[6] Action Matching: Learning Stochastic Dynamics From Samples (Neklyudov et al. 2022) [Paper](https://arxiv.org/abs/2210.06662)

[7] Riemannian Flow Matching on General Geometries (Chen et al. 2023) [Paper](https://arxiv.org/abs/2302.03660)

[8] Conditional Flow Matching: Simulation-Free Dynamic Optimal Transport (Tong et al. 2023) [Paper](https://arxiv.org/abs/2302.00482)

[9] Minimizing Trajectory Curvature of ODE-based Generative Models (Lee et all. 2023) [Paper](https://arxiv.org/abs/2301.12003)

### Why choose ODE-based model instead of SDE-based diffusion models or Denosing diffusion models?

Because ODE-based model is simpler in theory and implementation, it has become very popular recently.

### Why artifacts and glitches exist in the generated samples?
Since Rectified Flow is a proposed approach based on image generation, it may need to be modified or improved for speech tasks. On the other hand, glitches in image generation (e.g., unnatural hands) are less likely to affect the overall image quality, but glitches in speech are naturally easy to capture perceptually.

### How to improve Rectified Flow?
[5] proposed that the loss function of Rectified Flow is biased and [9] proposed that Rectified Flow estimates the upper bound of the degree of intersection of the independent coupling but does not really minimize it, and improvements based on the loss function might improve its quality

## Reference
https://github.com/gnobitab/RectifiedFlow

