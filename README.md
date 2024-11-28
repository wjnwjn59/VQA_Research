# Visual Question Answering Research

## Install packages

### Git LFS

```bash
sudo apt-get install git-lfs
```

### Python

```bash
pip install -r requirements.txt
```

## Download datasets

```bash
cd dataset
python download.py
```

## Scripts
```bash
python3 train.py --seed 59 --is_text_augment False --is_img_augment False --use_dynamic_thresh False
python3 train.py --seed 59 --dataset_name openvivqa --is_text_augment True --n_text_paras 1 --n_text_para_pool 20 --text_para_thresh 0.6 --use_dynamic_thresh False
python3 train.py --seed 59 --is_text_augment True --n_text_paras 1 --n_text_para_pool 30 --text_para_thresh 0.6 --is_img_augment False --use_dynamic_thresh False
python3 train.py --seed 59 --is_img_augment True --n_img_augments 1 --img_augment_thresh 0.2 --is_text_augment False  --use_dynamic_thresh False
python3 train.py --seed 59 --use_dynamic_thresh True --is_text_augment True --n_text_paras 1 --n_text_para_pool 30 --n_text_para_pool 30 --text_para_thresh 0.6  --is_img_augment False
```

## Description

This repo contains the survey, source and re-implementation code of several methods related to the Visual Question Answering task.

## To-do list

- [x] Skeleton source
- [] Benchmarks

### General Methods

- [x] CNN+LSTM Classifier
- [x] CNN+LSTM+Attention Classifier
- [x] Transformers Encoder-Decoder
- [x] CNN+BERT Classifier
- [x] ViT+LSTM Classifier
- [x] BERT+ViT Classifier
- [x] BLIP Finetuning

### ViVQA Methods

- [x] PhoBERT+ViT Classifier
- [x] BARTPpho+ViT Classifier
- [x] BARTpho+BEiT Classifier

## Benchmarks

### Vietnamese VQA

|                        Dataset                         |                   Description                    |              Size               |
| :----------------------------------------------------: | :----------------------------------------------: | :-----------------------------: |
|     [ViTextVQA](https://arxiv.org/abs/2404.10652)      |     Vietnamese Text Comprehension in Images      | 16.762 images, 50.342 QA pairs  |
|       [EVJVQA](https://arxiv.org/pdf/2302.11752)       |      Multilingual Visual Question Answering      |  4.879 images, 33.790 QA pairs  |
|    [ViOCRVQA](https://arxiv.org/html/2404.18397v1)     |   Vietnamese Optical Character Recognition VQA   | 28.282 images, 123.781 QA pairs |
|     [OpenViVQA](https://arxiv.org/abs/2305.04183)      | Open-domain Vietnamese Visual Question Answering | 11.199 images, 37.914 QA pairs  |
|      [ViCLEVR](https://arxiv.org/abs/2310.18046)       |           Vietnamese Visual Reasoning            | 26.216 images, 30.000 QA pairs  |
| [ViVQA](https://aclanthology.org/2021.paclic-1.72.pdf) |       Vietnamese Visual Question Answering       | 10.328 images, 15.000 QA pairs  |

