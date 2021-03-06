clc;
clear;
trainData = loadMNISTImages('mnist/train-images-idx3-ubyte');
trainLabels = loadMNISTLabels('mnist/train-labels-idx1-ubyte');

trainLabels(trainLabels == 0) = 10; 
numClasses = length(unique(trainLabels));
rbfHiddenSize = 10;
autoencoderHiddenSize = 200;

settings.Sigmavalue = 10;
settings.lambda = 3e-3;         
settings.beta = 0.001;
settings.sparsityParam = 0.05;
settings.kmeansItera = 2;
settings.autoencoderMinibatch = 'on';
settings.autoencoderBatchNum = 20;
settings.autoencoderMaxepoch = 1000;
settings.autoencoderOptions.Method = 'lbfgs'; 
settings.autoencoderOptions.maxIter = 100;
settings.autoencoderOptions.display = 'on';
settings.fineTuningOptions.Method = 'lbfgs'; 
settings.fineTuningOptions.maxIter = 1000;
settings.fineTuningOptions.display = 'on';
[netConfig] = doubleLayerRbfAutoencoder(trainData,rbfHiddenSize,autoencoderHiddenSize,settings);
[featuresTransTraining] = dataMapping(trainData, netConfig);
%% the toplayer
fprintf('\n');
fprintf('Training the classifier ... \n');
fprintf('\n')

softmaxOptions.maxIter = 1;
lambdaSoftmax = 0.001;
[inputSizeSoftMax, ~] = size(featuresTransTraining);
[softmaxModel] = softmaxTrain(inputSizeSoftMax, numClasses, lambdaSoftmax, featuresTransTraining, trainLabels, softmaxOptions);

%% testing
fprintf('\n');
fprintf('Start the prediction ... \n');
fprintf('\n')
testData  = loadMNISTImages('mnist/t10k-images-idx3-ubyte');
testLabels =  loadMNISTLabels('mnist/t10k-labels-idx1-ubyte');

[featuresTransTesting] = dataMapping(testData, netConfig); 
[pred] = softmaxPredict(softmaxModel, featuresTransTesting);

acc = mean(testLabels(:) == pred(:));
fprintf('Accuracy: %0.3f%%\n', acc * 100);
 
