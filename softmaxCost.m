function [cost, grad] = softmaxCost(theta, numClasses, inputSize, lambda, data, labels)

% numClasses - the number of classes 
% inputSize - the size N of the input vector
% lambda - weight decay parameter
% data - the N x M input matrix, where each column data(:, i) corresponds to
%        a single test set
% labels - an M x 1 matrix containing the labels corresponding for the input data
%

% Unroll the parameters from theta
theta = reshape(theta, numClasses, inputSize);

numCases = size(data, 2);

groundTruth = full(sparse(labels, 1:numCases, 1));

cost = 0;

thetagrad = zeros(numClasses, inputSize);

%% 
%  Instructions: Compute the cost and gradient for softmax regression.
%                You need to compute thetagrad and cost.
%                The groundTruth matrix might come in handy.
%*********calculate cost**************************************

%lable_prob = sigmoid(theta*data);
%[~,MaxIndex] = max(lable_prob,[],1); 
%groundResults = full(sparse(MaxIndex,1:numCases,1));
%find the results of the classfier: the max prob one 

%groundCorrect = groundResults.*groundTruth;s
groundCorrect = groundTruth;
%find the correct results


groundNormal = repmat(sum(exp(theta*data),1),numClasses,1);
%normal term as denominator, also to make sure the sum prob =1
groundProb = exp(theta*data);
% term as numerator
regterm = (lambda/2)*sum(sum(theta.^2));
% regulaztion term to makesure the function is convex
cost =(-1/numCases)*sum(sum(groundCorrect.*log(groundProb./groundNormal)))+regterm;

%********* end caculate cost**********************************

%*********caculate gradient***********************************
thetagrad= (-1/numCases)*((groundCorrect-groundProb./groundNormal)*data')+lambda*theta;
%*************end gradient**********************************
% ------------------------------------------------------------------
% Unroll the gradient matrices into a vector for minFunc
grad = [thetagrad(:)];
end
