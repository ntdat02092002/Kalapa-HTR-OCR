# Kalapa - Vietnamese Handwritten Text Recognition
Building a lightweight OCR model to solve the problem of recognizing handwritten address in Vietnam

https://challenge.kalapa.vn/portal/handwritten-vietnamese-text-recognition/overview

## 0. Pipeline (optional)
### Stage1
Generate easy data with printed fonts and train model from this (this data use only once)
### Stage2
Generate harder data with hard fonts and finetune model from this (this data use only once)
### Stage3
Generate data with selected handwritten fonts and finetune model from this generated data + training data from competition
### Finetune, finetune and continuously finetune
Generate more data (little data each finetune stage) and finetune model from previous data + this generated data (concatenate)

## 1. Datasets
### 1.1 Prepare data

We use tool [VietNamese-OCR-DataGenerator](https://github.com/docongminh/VietNamese-OCR-DataGenerator) to generate synthetic data for training and fine-tuning. 

To use this tool, you need the labels, fonts, and background images. 
For preparing labels, we use labels from the training set and test set of the [Cinamon](https://github.com/pbcquoc/vietnamese_ocr/blob/master/README.md) set published in the CINAMON AI MARATHON competition along with labels from the training set of the organizers of this task.
    
For preparing fonts, we choose some fonts from Google for Vietnamese at [here](https://fonts.google.com/?subset=vietnamese&noto.script=Latn).
    
With background images, we use the background images which are available in the repository above.
    
### 1.2 Download data

After that, we generated 5 synthtic data for training and fine-tuning (arrange in order of generation as described in the pipeline section):

- images_no_style([link](https://drive.google.com/file/d/1_MzAYl_8pPqhOE1WSUO68Wl1d8vrzFep/view?usp=sharing))

- images_style ([link](https://drive.google.com/file/d/1AOlXamwoEeFx5PxqJTBAgSV4sMP_BtvE/view?usp=sharing))

- images_htr ([link](https://drive.google.com/file/d/1jb6dSXOcBzy40bacAF341IXQ87D-RmB1/view?usp=sharing))
    
- images_kalapa+cinamon ([link](https://drive.google.com/file/d/1Is1-Ao12F-uqK1l9DwBKRWNRHMxWyCYJ/view?usp=sharing))
    
- images_2k ([link](https://drive.google.com/file/d/1WsKy7dg71AwyRRJGqiNFTm-DF0DTwRDu/view?usp=sharing))
    
<!-- - images_3k ([link](https://drive.google.com/file/d/18JzsuYjMGm_IuUGhfWlKnL1E_XwcDFbT/view?usp=sharing)) -->

### 1.3 Split data for training
To train the model, we need to divide the data into two sets: train set and valid set. Regarding the division principle, we will use `10%` of the data to make the valid set, the rest is the training set. To ensure consistency, we will set `random.seed(0)` and use `random` to randomly select the valid set without repetition.

To divide data, we have provided code to divide data in `split_data.py`.
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
File .pth contains state_dict, optimizer, and other parameters, to export only model to use [Kaggle notebook](https://www.kaggle.com/code/ntdat02092002/kalapa-submit-private), use file `save_model.py` (change paths inside this file)




