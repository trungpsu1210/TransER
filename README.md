# TransER: Hybrid Model and Ensemble-based Sequential Learning for Non-homogenous Dehazing #

This code originally is proposed for TransER for single image dehazing. TransER consists of two separate deep neural networks which are TransConv Fusion Dehaze (TFD) model in Stage I and Lightweight Ensemble Reconstruction (LER) network in Stage II. The proposed method is inspired by vision transformer, dehazing fusion network, ensemble learning, and knowledge distillation

## Illustration of TransER ##

![alt text](https://github.com/trungpsu1210/TransER/blob/main/Overal_architecture.png)

## Structure of the code ##

1. First Classifier - Bi-Conv-LSTM and Second Classifier - Domain Enrich: Two proposed methods, each folders will have
* model.py: proposed model
* create_HDF5.py: convert all the data and label to H5py files
* preprocessing_dataloader.py: preprocessed data
* train.py: train the model
* test.py: test the performance
2. Checkpoint, H5, Results: save all the corresponding files to here
3. pytorch-msssim: designed the msssim loss function
4. Visualization.ipynb: code for drawing the figures with latex fonts (bar, chart, line, confusion matrix,...) used for papers
5. utils.py: useful function using in the code

## Requirments ##

Pytorch 1.9.0

Python 3.8

Deep learning libraries/frameworks: OpenCV, HDF5, TensorBoard,Pandas,...

To run the code, make sure all the files are in the corresponding folders
