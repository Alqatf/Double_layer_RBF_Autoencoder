function [pred] = softmaxPredict(softmaxModel, data)

% softmaxModel - model trained using softmaxTrain
% data - the N x M input matrix, where each column data(:, i) corresponds to
%        a single test set
%
% Your code should produce the prediction matrix 
% pred, where pred(i) is argmax_c P(y(c) | x(i)).
 
% Unroll the parameters from theta
theta = softmaxModel.optTheta;  % this provides a numClasses x inputSize matrix
pred = zeros(1, size(data, 2));
%  Instructions: Compute pred using theta assuming that the labels start 
%                from 1.
lable_prob = sigmoid(theta*data);

[~,MaxIndex] = max(lable_prob,[],1); 

pred = MaxIndex;

% ---------------------------------------------------------------------

end


function sigm = sigmoid(x)
   sigm = 1 ./ (1 + exp(-x));
end
function sigmInv = sigmoidInv(x)
    sigmInv = sigmoid(x).*(1-sigmoid(x));
end
