  %data = rand(6,20);
  %[featurenum,samplenum] = size(data);
  %hiddenSize = 5;
  %visibleSize = featurenum;
  %theta = initializeParameters(hiddenSize, visibleSize);
  %sparsityParam = 0.05; 
  %desired average activation of the hidden units.
  %lambda = 3e-3;         
  %weight decay parameter       
  %beta = 5;              
  %weight of sparsity penalty term       
  %epsilon = 0.1;
  %K=1;
  %subFeatureNum = [2,2,2];
  
% [cost,grad] = SplitSparseAutoencoderCost(theta, visibleSize, hiddenSize, lambda, sparsityParam, beta, data,subFeatureNum,K);
% Instructions:

%%
clc;
clear;
data = rand(8,20);
[featureNum,sampleNum] = size(data);
lambda = 3e-3;         
beta = 0.001; 

sparsityParam = 0.05; 
inputSize = featureNum;
rbfHiddenSize = 4;
autoencoderHiddenSize =3;
visibleSize = featureNum;

 autoencoderTheta= initializeParameters(rbfHiddenSize, autoencoderHiddenSize, visibleSize);
 r  = sqrt(6) / sqrt(rbfHiddenSize+visibleSize+1); 
rbfCentroids = rand(rbfHiddenSize, inputSize) * 2 * r - r;
sigma = [1.2,1.2,1.2,1.2];
theta=[autoencoderTheta;rbfCentroids(:);sigma(:)];
%theta=[autoencoderTheta;rbfCentroids(:)];
[cost,grad] = doubleLayerRbfAutoencoderCost(theta, inputSize, rbfHiddenSize,autoencoderHiddenSize,... 
                             visibleSize,lambda, beta,sparsityParam, data);


numGrad = computeNumericalGradient( @(x)doubleLayerRbfAutoencoderCost(x, inputSize, rbfHiddenSize,autoencoderHiddenSize,... 
                             visibleSize,lambda, beta,sparsityParam, data), theta);
%%

%%
    diff = norm(numGrad-grad)/norm(numGrad+grad);
    disp(diff);
    if diff < 1e-8,
        disp ('OK')
    else
        disp ('Difference too large. Check your gradient computation again')
    end