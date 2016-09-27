function [featureTrans] = dataMapping( features, netConfig)
%DATAMAPPING Summary of this function goes here
%   Detailed explanation goes here
[~,sampleNum]=size(features);
theta = netConfig.theta; 
inputSize = netConfig.inputSize;
rbfHiddenSize = netConfig.rbfHiddenSize;
autoencoderHiddenSize = netConfig.autoencoderHiddenSize;
visibleSize = netConfig.visibleSize;

%% recover theta

W1 = reshape(theta(1:autoencoderHiddenSize*rbfHiddenSize), autoencoderHiddenSize,  rbfHiddenSize);
W2 = reshape(theta(autoencoderHiddenSize* rbfHiddenSize+1:autoencoderHiddenSize*(rbfHiddenSize+visibleSize)), visibleSize,  autoencoderHiddenSize);
b1 = theta(autoencoderHiddenSize*(rbfHiddenSize+visibleSize)+1:autoencoderHiddenSize*(rbfHiddenSize+visibleSize)+autoencoderHiddenSize);
b2 = theta(autoencoderHiddenSize*(rbfHiddenSize+visibleSize+1)+1:autoencoderHiddenSize*(rbfHiddenSize+visibleSize+1)+visibleSize);
rbfCentroids = reshape(theta(autoencoderHiddenSize*(rbfHiddenSize+visibleSize+1)+visibleSize+1:autoencoderHiddenSize*(rbfHiddenSize+visibleSize+1)...
                                                                 +visibleSize+rbfHiddenSize*inputSize), rbfHiddenSize, inputSize);
sigma = theta(autoencoderHiddenSize*(rbfHiddenSize+visibleSize+1)+visibleSize+rbfHiddenSize*inputSize+1:end);

%% recover data
for i = 1:rbfHiddenSize  % calculate the output node by nodeb
    c_vector = rbfCentroids(i,:); % get the center of this node
    c_matrix = repmat(c_vector,[sampleNum,1]);
    diff =  features - c_matrix';
    distance(i,:) = (arrayfun(@(x)(sum(diff(:,x).^2)),1:size(diff,2)))/(2*(sigma(i))^2);
    clear diff;
    clear c_matrix;
end
rbfLayerOutputFeatures = exp(-distance);
inputAutoencoderLayer1 = W1*rbfLayerOutputFeatures +repmat(b1,1,sampleNum);
outputAutoencoderLayer1 = sigmoid(inputAutoencoderLayer1);
featureTrans = outputAutoencoderLayer1; 
end


function sigm = sigmoid(x)
   sigm = 1 ./ (1 + exp(-x));
end
function sigmInv = sigmoidInv(x)

    sigmInv = sigmoid(x).*(1-sigmoid(x));
end