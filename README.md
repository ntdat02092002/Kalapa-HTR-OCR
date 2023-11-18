# Kalapa - Vietnamese Handwritten Text Recognition
Building a lightweight OCR model to solve the problem of recognizing handwritten address in Vietnam

https://challenge.kalapa.vn/portal/handwritten-vietnamese-text-recognition/overview

## 0. Pipeline (optional)
### Stage1
Generate easy data with printed fonts and train model from this 
### Stage2
Generate harder data with hard fonts and finetune model from this
### Stage3
Generate data with selected handwritten fonts and finetune model from this + data from competition
### Finetune, finetune and continuously finetune
Generate more data (little data each finetune stage) and finetune model from previous data + this

## 1. Download data
All our generated data here: ...updating...

We use tool <place_holder> to generate all this data

## 2. Train model
### 2.1 Setup environment
```bash
git clone https://github.com/ntdat02092002/Kalapa-HTR-OCR.git
cd ./Kalapa-HTR-OCR
pip install -r requirements.txt
```
### 2.2. Set config
See config in config.yaml file and change save_path, dataset path (data name, data dicrectory and annotation file) and other fields if you want

### 2.3. Training
#### Training from scratch
```bash
python train.py --config=./config.yaml
```
#### Finetune model
```bash
python train.py --config=./config.yaml --ckpt_pth="path/to/file/.pth/to/finetune"
```

## 3. Inference
```bash
python predict.py --config='path/to/config/file/get/from/training/stage' --weight='path/to/.pth/file/get/from/training/stage' --directory='path/to/folder/contain/images' --output='where/to/save/output'
# example:
# python predict.py --config=output/kalapa_vgg11_finetune/checkpoints/config.yml --weight=output/kalapa_vgg11_finetune/checkpoints/vgg_seq2seq.pth --directory=/workplace/datasets/Kalapa/OCR/public_test/images
# output default is ./predics/
```

## 4. Save model
File .pth contains state_dict, optimizer, and other parameters, to export only model to use Kaggle notebook, use file "save_model.py" (change paths inside this file)




