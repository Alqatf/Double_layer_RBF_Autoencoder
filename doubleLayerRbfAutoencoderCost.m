function [cost,grad] = doubleLayerRbfAutoencoderCost(theta, inputSize, rbfHiddenSize,autoencoderHiddenSize, visibleSize,lambda, beta,sparsityParam,features)

[~,sampleNum]=size(features);
W1 = reshape(theta(1:autoencoderHiddenSize*rbfHiddenSize), autoencoderHiddenSize,  rbfHiddenSize);
W2 = reshape(theta(autoencoderHiddenSize* rbfHiddenSize+1:autoencoderHiddenSize*(rbfHiddenSize+visibleSize)), visibleSize,  autoencoderHiddenSize);
b1 = theta(autoencoderHiddenSize*(rbfHiddenSize+visibleSize)+1:autoencoderHiddenSize*(rbfHiddenSize+visibleSize)+autoencoderHiddenSize);
b2 = theta(autoencoderHiddenSize*(rbfHiddenSize+visibleSize+1)+1:autoencoderHiddenSize*(rbfHiddenSize+visibleSize+1)+visibleSize);
rbfCentroids = reshape(theta(autoencoderHiddenSize*(rbfHiddenSize+visibleSize+1)+visibleSize+1:autoencoderHiddenSize*(rbfHiddenSize+visibleSize+1)...
                                                                 +visibleSize+rbfHiddenSize*inputSize), rbfHiddenSize, inputSize);
sigma = theta(autoencoderHiddenSize*(rbfHiddenSize+visibleSize+1)+visibleSize+rbfHiddenSize*inputSize+1:end);

cost = 0;
W1grad = zeros(size(W1)); 
W2grad = zeros(size(W2));
b1grad = zeros(size(b1)); 
b2grad = zeros(size(b2));
rbfCentroidsgrad = zeros(size(rbfCentroids));
sigmagrad = zeros(size(sigma));
                                                             
%% Forwarding : RBF layer                                               
for i = 1:rbfHiddenSize  % calculate the output node by nodeb
    c_vector = rbfCentroids(i,:); % get the center of this node
    c_matrix = repmat(c_vector,[sampleNum,1]);
    diff =  features - c_matrix';
    distance(i,:) = (arrayfun(@(x)(sum(diff(:,x).^2)),1:size(diff,2)))/(2*(sigma(i))^2);
    clear diff;
    clear c_matrix;
end
rbfLayerOutputFeatures = exp(-distance);
%% Forwarding : Autoencoder layer
inputAutoencoderLayer1 = W1*rbfLayerOutputFeatures +repmat(b1,1,sampleNum);
outputAutoencoderLayer1 = sigmoid(inputAutoencoderLayer1);
inputAutoencoderLayer2 = W2*outputAutoencoderLayer1+repmat(b2,1,sampleNum);
outputAutoencoderLayer2 = sigmoid(inputAutoencoderLayer2);

cost_main = (0.5/sampleNum)*sum(sum((features - outputAutoencoderLayer2).^2));%error term in the cost
weight_decay = 0.5*(sum(sum(W1.^2))+sum(sum(W2.^2)));%the weigh dacay

rho = (1/sampleNum)*sum(outputAutoencoderLayer1,2);
Regterm =  sum(sparsityParam.*log(sparsityParam./rho)+(1-sparsityParam).*log((1-sparsityParam)./(1-rho)));%regularization term

cost =cost_main +lambda*weight_decay+beta*Regterm;
%% Backpropagation for grad

errortermAutoencoderLayer2 = -(features - outputAutoencoderLayer2).*sigmoidInv(inputAutoencoderLayer2);
reg_grad =beta*(-sparsityParam./rho+(1-sparsityParam)./(1-rho));

errortermAutoencoderLayer1=(W2'*errortermAutoencoderLayer2+repmat(reg_grad,1,sampleNum)).*sigmoidInv(inputAutoencoderLayer1);

W1grad = W1grad+errortermAutoencoderLayer1*rbfLayerOutputFeatures';
W1grad = (1/sampleNum).*W1grad+lambda*W1;

W2grad = W2grad + errortermAutoencoderLayer2*outputAutoencoderLayer1';
W2grad = (1/sampleNum).*W2grad+lambda*W2;

b1grad = b1grad+sum(errortermAutoencoderLayer1, 2);
b1grad = (1/sampleNum)*b1grad;

b2grad = b2grad+sum(errortermAutoencoderLayer2, 2);
b2grad = (1/sampleNum)*b2grad;

errortermRbfLayer = W1'*errortermAutoencoderLayer1.*rbfLayerOutputFeatures;
for j = 1: rbfHiddenSize
    
    c_vector = rbfCentroids(j,:); % get the center of this node
    c_matrix = repmat(c_vector,[sampleNum,1]);
    diff =  features - c_matrix';
    
    errordiff = errortermRbfLayer(j,:)*(diff)';
    rbfCentroidsgradUpdate(j,:) = errordiff/(sigma(j)^2);
    
    errorDiffsig = sum(errortermRbfLayer(j,:)*(diff.^2)');
    sigmagradUpdate(j,:) = errorDiffsig/(sigma(j)^3);
    
     clear diff;
    clear c_matrix;
end

rbfCentroidsgrad = rbfCentroidsgrad + rbfCentroidsgradUpdate;
rbfCentroidsgrad = (1/sampleNum).*rbfCentroidsgrad ;

sigmagrad = sigmagrad + sigmagradUpdate;
sigmagrad = (1/sampleNum).*sigmagrad;

grad = [W1grad(:) ; W2grad(:) ; b1grad(:) ; b2grad(:); rbfCentroidsgrad(:);sigmagrad(:)];
%grad = [W1grad(:) ; W2grad(:) ; b1grad(:) ; b2grad(:); rbfCentroidsgrad(:)];
end



%-------------------------------------------------------------------
% Here's an implementation of the sigmoid function, which you may find useful
% in your computation of the costs and the gradients.  This inputs a (row or
% column) vector (say (z1, z2, z3)) and returns (f(z1), f(z2), f(z3)). 

function sigm = sigmoid(x)
   sigm = 1 ./ (1 + exp(-x));
end
function sigmInv = sigmoidInv(x)
    sigmInv = sigmoid(x).*(1-sigmoid(x));
end

