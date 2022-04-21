# Setups

## Environment: 

CPU: Intel 8- Core i7-11800

GPU: RTX 3060

cudnn-11.2-windows-x64-v8.1.0.77

cuda_11.2.1_461.09_win10

## External libraries:

tensorflow-gpu==2.5.0

keras

transformers

matplotlib

pandas

sklearn

sentencepiece

# 1. Brief introduction

Sentiment levels of movie comments can be scored by sentences or phrases. The purpose of this project is to explore the inference power of four transformer-based models which have been pre-trained on self-supervised learning tasks. 

Specifically, in this work, the split phrases are classified into five levels of emotions by transfer learning with BERT, RoBERTa, DistilBERT and XLNet.

Click [Kaggle Competition Link](https://www.kaggle.com/competitions/sentiment-analysis-on-movie-reviews) for more details.

# 2. Organization of the files

## 2.1. Dataset

**File "./Datasets/train.csv" :**

Even though the train/test split has been preserved for the purposes of benchmarking, the labels of the test set are not available. Therefore, the original training dataset with 156,060 phrases was divided into a new training set, validation set, and testing set in terms of proportion 8:1:1. The restructured training set has 140,454 phrases.

The sentiment labels and numbers are within restructured training set:

0 - negative-6371

1 - somewhat negative-24545

2 - neutral-71624

3 - somewhat positive-29626

4 - positive-8288

## 2.2. Folder "Function"

This folder divides the entire project process into data preprocessing, model, train, and test procedures, waiting for calling by "main.py".

**All functions within this folder with detailed comments**

### 2.2.1. File "data_preprocessing.py"

1. Split into the train(validation within it) and test

2. Extract useful features and labels

3. Set your model output as categorical and save it in the new label column

4. Split the training set into training and validation dataset

5. tokenizer function (in "train.py")

   Return the training, validation and testing datasets.

### 2.2.2. File "model.py"

Define model structures of BERT, RoBERTa, DistilBERT and XLNet.

### 2.2.3. File "train.py"

Define the training pipelines with an optimizer, loss, and other hyper-parameters adjustable.

Return the model after training and logs history to analyze training convergence.

### 2.2.4. File "test.py"

This part evaluates the model performance on the testing dataset containing:

1. Classification Reports.

2. Confusion Matrix figures.

3. Multi-class ROC figures.

4. Globally micro ROC figures.

## 2.3. Other Folders

Folder  "images" save the training and testing figures, as well as model structures.

Folder  "model_logs" saves the training logs history.

Folder  "model_trained" saves the model after training.

Folder  "submission" saves the .csv file using the same format as the competition.

# 3. Run the code

## 3.1. Run the code on "main.py"

Notes. It will run all steps of this project by calling the functions in the folder "function".

## 3.2. Run the code on "main.ipynb"

Notes. This contains more detailed work than in "main.py", including the EDA process at the beginning, open in Jupyter for running step by step.

### All trained models employed in the report can be obtained in the  [GOOGLE DRIVE](https://drive.google.com/drive/folders/1STGwHfYwuoKsOVNKbPQgW3_PtNOjfgqk?usp=sharing) link

### DO NOT CHANGE THE FILE LEVEL



























