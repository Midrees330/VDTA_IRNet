# VDTA_IRNet
Variational Disentanglement for Task-Agnostic Image Restoration
# Requirement
- Python 3.11
- Pytorch 2.0
- CUDA 11.7
- MATLAB R2023b
# Datasets
- AISTD (ISTD+) [link](https://github.com/cvlab-stonybrook/SID)
- SRD [Training](https://drive.google.com/file/d/1W8vBRJYDG9imMgr9I2XaA13tlFIEHOjS/view) [Testing](https://drive.google.com/file/d/1GTi4BmQ0SJ7diDMmf-b7x2VismmXtfTo/view) [Mask](https://uofmacau-my.sharepoint.com/:u:/g/personal/yb87432_um_edu_mo/EZ8CiIhNADlAkA4Fhim_QzgBfDeI7qdUrt6wv2EVxZSc2w?e=wSjVQT) (detected by [DHAN](https://github.com/vinthony/ghost-free-shadow-removal))
# Pretrained models
The corresponding pretrained models: 
# Test the model
You can directly test the performance of the pre-trained model as follows:
Modify the paths to dataset and pre-trained model. You need to modify the following path in the `test.py` or run
- python test.py --load [checkpoint numbers, e.g 1160]
# Train
1. Download datasets and set the following structure

    ```
    -- AISTD_Dataset
       |-- train
       |   |-- train_A  # shadow image
       |   |-- train_B  # shadow mask (not used)
       |   |-- train_C  # shadow-free GT
       |
       |-- test
           |-- test_A  # shadow image
           |-- test_B  # shadow mask (not used)
           |-- test_C  # shadow-free GT

    -- SRD_Dataset
       |-- train
       |   |-- train_A  # shadow image
       |   |-- train_B  # shadow mask (not used)
       |   |-- train_C  # shadow-free GT
       |
       |-- test
           |-- test_A  # shadow image
           |-- test_B  # shadow mask (not used)
           |-- test_C  # shadow-free GT

    -- BSD_Dataset
       |-- BSDdataset
       |   |-- train  # denoise image
       |
       |-- test (BSD68)
           |-- noisy15  # noisy image
           |-- noisy25  # noisy image
           |-- noisy50  # noisy image
           |-- original  # clean image

    -- RAIN_Dataset
       |-- train
       |   |-- train_A  # rain image
       |   |
       |   |-- train_B  # rain-free GT
       |
       |-- test
           |-- test_A  # rain image
           |
           |-- test_B  # rain-free GT
# Evaluation
The results reported in the paper are calculated by the `matlab` script used in previouse method: https://github.com/hhqweasd/G2R-ShadowNet/blob/main/evaluate.m
# Testing results
The testing results on dataset  AISTD (ISTD+), SRD, BSD68, rain100L are:
# Contact
If you have any questions, please contact idreeskhan045@gmail.com/ huangying@cqupt.edu.cn
