# TransER: Hybrid Model and Ensemble-based Sequential Learning for Non-homogenous Dehazing #

This code originally is proposed for TransER for single image dehazing. TransER consists of two separate deep neural networks which are TransConv Fusion Dehaze (TFD) model in Stage I and Lightweight Ensemble Reconstruction (LER) network in Stage II. The proposed method is inspired by vision transformer, dehazing fusion network, ensemble learning, and knowledge distillation

## Illustration of TransER ##

![alt text](https://github.com/trungpsu1210/TransER/blob/main/Overal_architecture.png)

## Structure of the code ##

1. models: Folders for proposed networks (TFD, TRN, and LER)
2. train_xxx.py: to train the models in multiple steps
3. test.py: to test the quantitative results
4. predict.py: tp generate the haze-free images
5. data_loader.py: to load the pair input and output data or single images
6. perceptual.py: vgg loss function
7. configs.json: configuration parameter to train, test, and predict
8. utils: useful function using in the code
9. empatches: function to extract and merge multiple patches

## Requirments ##

Pytorch 1.10.2

Python 3.7

cudatoolkit=11.3

Deep learning libraries/frameworks: OpenCV, HDF5, TensorBoard, timm, torchvision, pytorch_msssim...

To run the code, make sure all the files are in the corresponding folders
