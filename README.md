Here’s a modified version of the README that aligns with the requirements for your course project:

---

# Light-SERNet: A Lightweight Fully Convolutional Neural Network for Speech Emotion Recognition

This repository contains the TensorFlow 2.x implementation of the paper ["Light-SERNet: A lightweight fully convolutional neural network for speech emotion recognition"](https://arxiv.org/abs/2110.03435), accepted in ICASSP 2022.

## Overview

Light-SERNet is an efficient and lightweight fully convolutional neural network (FCNN) for speech emotion recognition, designed to work on systems with limited hardware resources. The architecture uses three parallel paths with different filter sizes to extract feature maps. This method ensures that deep convolutional blocks can extract high-level features, making the model competitive despite having a smaller size than state-of-the-art models.

The model achieves superior performance on two benchmark datasets, **IEMOCAP** and **EMO-DB**, with a reduced computational footprint.

---

## Demo

You can test the model on the EMO-DB dataset using Google Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AryaAftab/LIGHT-SERNET/blob/master/Demo_Light_SERNet.ipynb)

---

## Requirements

To set up the environment and run the codes, first install the dependencies using the `requirements.txt` file:

### Required Libraries

- TensorFlow >= 2.3.0
- NumPy >= 1.19.2
- Tqdm >= 4.50.2
- Matplotlib >= 3.3.1
- Scikit-learn >= 0.23.2

### Installation

```bash
$ git clone https://github.com/AryaAftab/LIGHT-SERNET.git
$ cd LIGHT-SERNET/
$ pip install -r requirements.txt
```

---

## Dataset

Download the **[EMO-DB](http://emodb.bilderbar.info/download/download.zip)** and **[IEMOCAP](https://sail.usc.edu/iemocap/iemocap_release.htm)** datasets (IEMOCAP requires permission to access). After downloading, extract the datasets into the `./data` directory.

For using **IEMOCAP**, follow the issue in [#3](../../issues/3).

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

### Example: Training on EMO-DB dataset

```bash
$ python train.py -dn "EMO-DB" \
                  -id 3 \
                  -at "all" \
                  -ln "focal" \
                  -v 1 \
                  -it "mfcc" \
                  -c "disk" \
                  -m false
```

### Example: Training on IEMOCAP dataset

```bash
$ python train.py -dn "IEMOCAP" \
                  -id 7 \
                  -at "impro" \
                  -ln "cross_entropy" \
                  -v 1 \
                  -it "mfcc" \
                  -c "disk" \
                  -m false
```

To run all experiments, simply execute the `run.sh` script:

```bash
sh run.sh
```

---

## MFCC Feature Fusing (New Feature)

Light-SERNet allows you to run the model without TensorFlow by incorporating the MFCC feature extractor as a layer. The trained model can be exported in TensorFlow Lite format, and it takes raw audio input as a vector.

To enable this feature during training:

```bash
$ python train.py -dn "EMO-DB" \
                  -id 3 \
                  -m True
```

---

## Results

Training results, including the confusion matrix and performance metrics, will be saved in the `./result` folder, and the best model will be stored in the `./model` folder.

---

## Citation

If you find this code useful for your research, please consider citing the original paper:

```bibtex
@inproceedings{aftab2022light,
  title={Light-SERNet: A lightweight fully convolutional neural network for speech emotion recognition},
  author={Aftab, Arya and Morsali, Alireza and Ghaemmaghami, Shahrokh and Champagne, Benoit},
  booktitle={ICASSP 2022-2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={6912--6916},
  year={2022},
  organization={IEEE}
}
```

---

### PDF Report (to be included)

Your final repository should also include a **PDF (not exceeding 2 pages)**, which will summarize the implementation, results, and dataset description. This can be prepared in LaTeX (similar to previous submissions).

---

This updated README structure clearly outlines the steps needed to set up and run the code, while also detailing how to train the model and test it on datasets. Make sure to include the `requirements.txt`, all the necessary code files, and the PDF summary when you submit the project.
