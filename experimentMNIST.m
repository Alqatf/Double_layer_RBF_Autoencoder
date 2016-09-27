clc;
clear;
trainData = loadMNISTImages('mnist/train-images-idx3-ubyte');
trainLabels = loadMNISTLabels('mnist/train-labels-idx1-ubyte');
trainLabels(trainLabels == 0) = 10; 
numClasses = size(unique(trainLable));
rbfHiddenSize = 30;
autoencoderHiddenSize = 200;

settings.Sigmavalue = 1.2;
settings.lambda = 3e-3;         
settings.beta = 0.001;
settings.sparsityParam = 0.05;
settings.kmeansItera = 1;
settings.autoencoderOptions.Method = 'lbfgs'; 
settings.autoencoderOptions.maxIter = 1;
settings.autoencoderOptions.display = 'on';
settings.fineTuningOptions.Method = 'lbfgs'; 
settings.fineTuningOptions.maxIter = 1;
settings.fineTuningOptions.display = 'on';
[netConfig] = doubleLayerRbfAutoencoder(trainData,rbfHiddenSize,autoencoderHiddenSize,settings);
[featuresTransTraining] = dataMapping( features, netConfig);
%% the toplayer
softmaxOptions.maxIter = 1;
lambdaSoftmax = 0.001;
[inputSizeSoftMax, ~] = size(featuresTransTraining);
[softmaxModel] = softmaxTrain(inputSizeSoftMax, numClasses, lambdaSoftmax, featuresTransTraining, trainLabels, softmaxOptions); 