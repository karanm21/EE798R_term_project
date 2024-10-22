# Light-SERNet

This repository contains the TensorFlow 2.x implementation of the paper ["Light-SERNet: A lightweight fully convolutional neural network for speech emotion recognition"](https://arxiv.org/abs/2110.03435), accepted in ICASSP 2022.

## Overview

Light-SERNet is an efficient and lightweight fully convolutional neural network (FCNN) for speech emotion recognition, designed to work on systems with limited hardware resources. The architecture uses three parallel paths with different filter sizes to extract feature maps. This method ensures that deep convolutional blocks can extract high-level features, making the model competitive despite having a smaller size than state-of-the-art models.

The model achieves superior performance on two benchmark datasets, **IEMOCAP** and **EMO-DB**, with a reduced computational footprint.

---

## Requirements

To set up the environment and run the codes, first install the dependencies using the `requirements.txt` file:

### Required Libraries

- TensorFlow-gpu >= 2.3.0
- NumPy >= 1.19.2
- Tqdm >= 4.50.2
- Matplotlib >= 3.3.1
- Scikit-learn >= 0.23.2

### Installation

```bash
$ git clone https://github.com/karanm21/EE798R_term_project.git
$ cd EE798R_term_project/
$ pip install -r requirements.txt
```

---

## Dataset

Download the **[EMO-DB](http://emodb.bilderbar.info/download/download.zip)** and **[IEMOCAP](https://sail.usc.edu/iemocap/iemocap_release.htm)** datasets (IEMOCAP requires permission to access). After downloading, extract the datasets into the `./data` directory.

---

## Training and Testing

You can train the model on your desired dataset by specifying the parameters as follows:

```bash
$ python train.py -dn {dataset_name} \
                  -id {input durations} \
                  -at {audio_type} \
                  -ln {cost function name} \
                  -v {verbose for training bar} \
                  -it {type of input (mfcc, spectrogram, mel_spectrogram)} \
                  -c {type of cache (disk, ram, None)} \
                  -m {fuse mfcc feature extractor in exported tflite model}
```


To run all experiments, simply execute the `run.sh` script:

```bash
sh run.sh
```

---


## Results

Training results, including the confusion matrix and performance metrics, will be saved in the `./result` folder, and the best model will be stored in the `./model` folder.

---
