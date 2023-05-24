# Semantic_Segmentation_UAVid

This project contains a full pipeline of training and inferring of UAVid dataset using three different models (SegFormer,UnetFormer,Feature Pyramid) models for image segmentation using Pytorch-Lightning. \
This project uses ensemble learning to achieve the task.

- I have used pytorch 2.0 and pytorch-lightning 2.0 as a framework to the networks.

## Install

You can install using the following commands:
```
conda create -n uavid python=3.9
conda activate uavid
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r ./requirements.txt
```
## Models 
This repository contains three architectures:

- SegFormer 
- UNetFormer
- Feature Pyramid Network 

You can add your own architecture by add a new config file in [config folder](config/), and add a Pytorch-Lightning Module in [lightning_module file](src/lightning/lightning_module.py)


Pretrained Weights of models on UAVid can be access from [Yandex Disk](https://disk.yandex.ru/d/AINrvKNrpEjjpQ)

## Data Preprocessing
You can download the dataset from [Kaggle](https://www.kaggle.com/dasmehdixtr/uavid-v1).

And then we need to **relabel** it and **split** it into patches of 1024*1024 each using: 

```
python utils/uavid_patch_split.py \
--input-dir "./uavid/uavid_train" \
--output-img-dir "./uavid/train/images" \
--output-mask-dir "./uavid/train/masks" \
--split-size-h 1024 --split-size-w 1024 \
--stride-h 1024 --stride-w 1024
```
Please do so for both train and val sets. 

## Training
For training, you only need to adjust your config file and pass it to train file

```-c``` the path of the config, use different **config** to train different models.

```
python3 train.py -c config_file
```

## Inference
The inference supposes that you pass one or more models with their correspondent weight in the ensemble learning. 
This is an example of running the three models together 

```-i``` input path 

```-c```  path of the config files of models. You may use one or more

```-w```  The weight for each config file (for each model) 

```-o``` output path 

```-b``` batch size

```-m``` Whether the set has or not masks (test set does not have masks to evaluate on)
```
python3 inference.py \
-i ./uavid/uavid_val \
-c ./config/Unetformer.py ./config/Segformer.py ./config/FPN.py \
-w 0.5 0.5 0.5 \
-o ./results/ \
-ph 1024 -pw 1024 \
-b 6 \
-m 
```

## Report

A small report of training process of the models can be seen [in WandD](https://wandb.ai/maliksyria/UAVid_Semantic/reports/Semantic-Segmentation-of-UAVid-Dataset--Vmlldzo0NDU1MTEy)

## TODO List
- Create Jupyter Notebook as a demo of the work
- Create docker
- Test Time Augmentation (TTA): full implementation for all models 