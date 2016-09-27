function [netConfig ] = doubleLayerRbfAutoencoder(features,rbfHiddenSize,autoencoderHiddenSize,settings )
%DOUBLELAYERRBFAUTOCODER Summary of this function goes here
%   Detailed explanation goes here
%% Calculate the RBF centers
sigmavalue = settings.Sigmavalue;
lambda = settings.lambda ;         
beta = settings.beta;
sparsityParam = settings.sparsityParam;

[featureNum,sampleNum]=size(features);
rbfCentorids = runkmeans(features',rbfHiddenSize,1);

sigma = repmat(sigmavalue,[1,rbfHiddenSize]);

for i = 1:rbfHiddenSize  % calculate the output node by nodeb
    c_vector = rbfCentorids(i,:); % get the center of this node
    c_matrix = repmat(c_vector,[sampleNum,1]);
    diff =  features - c_matrix';
    distance(i,:) = (arrayfun(@(x)(sum(diff(:,x).^2)),1:size(diff,2)))/(2*(sigma(i))^2);
    clear c_matrix
    clear diff;
end
rbfLayerOutputFeatures = exp(distance);

%dataOutputRbfLayer = 
%% Calculate the autoencoder layer
visibleSize = featureNum; 
autoencoderTheta = initializeParameters(rbfHiddenSize, autoencoderHiddenSize, visibleSize);

addpath minFunc/
options = struct;
options.Method = 'lbfgs'; 
options.maxIter = 1;
options.display = 'on';

optAutoencoderTheta = minFunc( @(p) sparseAutoEncoderLayerCost(p, ...
                                   rbfHiddenSize, autoencoderHiddenSize, visibleSize,...
                                   lambda, sparsityParam,beta, ...
                                    features,rbfLayerOutputFeatures), ...
                              autoencoderTheta , options);
%% Fine-tuning

optAllTheta = [optAutoencoderTheta; rbfCentorids(:);sigma(:)];
fineTuningOptions = struct;
fineTuningOptions.Method = 'lbfgs'; 
fineTuningOptions.maxIter = 1;
fineTuningOptions.display = 'on';
inputSize = featureNum;
finetuningTheta =  minFunc( @(p) doubleLayerRbfAutoencoderCost(p, inputSize, rbfHiddenSize,autoencoderHiddenSize,... 
                                                                                             visibleSize,lambda, beta,sparsityParam,features), ...
                                                                                             optAllTheta,fineTuningOptions);
netConfig.Theta = finetuningTheta;
end

