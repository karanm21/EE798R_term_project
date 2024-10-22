LIGHT-SERNET: Speech Emotion Recognition

Author: Karan Mundhra
Affiliation: Indian Institute of Technology Kanpur
Contact: karanm21@iitk.ac.in

Brief Overview

This project introduces LIGHT-SERNET, a lightweight fully convolutional neural network (CNN) designed for speech emotion recognition (SER) on devices with limited computational resources. The model processes speech using Mel-frequency cepstral coefficients (MFCCs) and employs three parallel convolutional paths to capture both time- and frequency-based features.

Despite having just a 0.88 MB model size, it achieves competitive performance on the IEMOCAP and EMO-DB datasets, with an unweighted average recall (UAR) of 70.78% and a weighted accuracy (WA) of 79.87% on IEMOCAP. The model's small footprint makes it suitable for low-power devices.

GitHub Repository: LIGHT-SERNET

Dataset Availability

The model is evaluated on two datasets:

IEMOCAP: A multimodal dataset with 12 hours of audio-visual recordings, featuring both scripted and improvised emotional speech. It contains 5,531 labeled utterances with emotions such as happiness, sadness, anger, and neutral.
EMO-DB: A German emotional speech dataset with 535 utterances across 7 emotion classes, recorded by 10 actors.
Kaggle versions of these datasets are available, but direct access to the original datasets is not provided.

Computational Effort Required

The model was trained on an NVIDIA Tesla V100 GPU for 300 epochs with a batch size of 32. Computational cost varies based on input length:

3-second inputs: Requires 322 MFLOPs and 1.6 MB of memory.
7-second inputs: Requires 760 MFLOPs and 3.8 MB of memory.
The model's lightweight nature is achieved by reducing the number of parameters and employing efficient convolutional paths, thus minimizing computational cost and peak memory usage (PMU).

Training Strategy

The model's training process involves:

Input Pipeline: Normalize audio signals, extract MFCCs, apply FFT and Mel-scale filter, and select 40 MFCC coefficients.
Body Part I:
Path 1: 9×1 CNN
Path 2: 1×11 CNN
Path 3: 3×3 CNN
Concatenation of Paths
Body Part II:
LFLBs (Locally Feature Learning Blocks)
Convolution
ReLU
Global Average Pooling (GAP)
Head:
Dropout (0.3)
Fully Connected Layer
Softmax Classification
Training is done using the Adam optimizer with an initial learning rate of 
1
0
−
4
10 
−4
 , decaying every 20 epochs after epoch 50. Regularization techniques include:

Batch normalization after each convolutional layer
Dropout rate of 0.3 before the softmax layer
L2 weight decay of 
1
0
−
6
10 
−6
  to prevent overfitting
The training runs for 300 epochs with a batch size of 32.
