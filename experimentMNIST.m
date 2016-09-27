clc;
clear;
trainData = loadMNISTImages('mnist/train-images-idx3-ubyte');
trainLabels = loadMNISTLabels('mnist/train-labels-idx1-ubyte');
trainLabels(trainLabels == 0) = 10; 

rbfHiddenSize = 30;
autoencoderHiddenSize = 200;

settings.Sigmavalue = 1.2;
settings.lambda = 3e-3;         
settings.beta = 0.001;
settings.sparsityParam = 0.05;

[netConfig ] = doubleLayerRbfAutoencoder(trainData,rbfHiddenSize,autoencoderHiddenSize,settings);


