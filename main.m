clc;
clear;

CIFAR_DIR='cifar-10-batches-mat/';

assert(strcmp(CIFAR_DIR, 'cifar-10-batches-mat/'), ...%strcmp相等时为1
       ['You need to modify kmeans_demo.m so that CIFAR_DIR points to ' ...
        'your cifar-10-batches-mat directory.  You can download this ' ...
        'data from:  http://www.cs.toronto.edu/~kriz/cifar-10-matlab.tar.gz']);
% for this data set, each data is 32-32 rgb 
%10classes

%% Configuration
addpath minFunc;
rfSize = 6;
numCentroids=1600;%
whitening=true;
numPatches = 100000;
CIFAR_DIM=[32 32 3];
% for this data set, each data is 32-32 rgb 
%% Load CIFAR training data
fprintf('Loading training data...\n');
f1=load([CIFAR_DIR '/data_batch_1.mat']);
f2=load([CIFAR_DIR '/data_batch_2.mat']);
f3=load([CIFAR_DIR '/data_batch_3.mat']);
f4=load([CIFAR_DIR '/data_batch_4.mat']);
f5=load([CIFAR_DIR '/data_batch_5.mat']);

trainX = double([f1.data; f2.data; f3.data; f4.data; f5.data]);%50000*3072
trainY = double([f1.labels; f2.labels; f3.labels; f4.labels; f5.labels]) + 1; % add 1 to labels!,变成1到10

%trainX = double([f1.data; f2.data; f3.data]);%50000*3072
%trainY = double([f1.labels; f2.labels; f3.labels]) + 1; % add 1 to labels
clear f1 f2 f3 f4 f5

% extract random patches
patches = zeros(numPatches, rfSize*rfSize*3);%400000*108
for i=1:numPatches
  if (mod(i,10000) == 0) 
      fprintf('Extracting patch: %d / %d\n', i, numPatches); 
  end
  
  r = random('unid', CIFAR_DIM(1) - rfSize + 1);%
  c = random('unid', CIFAR_DIM(2) - rfSize + 1);%
    
  %使用mod(i-1,size(trainX,1))是因为对每个图片样本，提取出numPatches/size(trainX,1)个patch
  patch = reshape(trainX(mod(i-1,size(trainX,1))+1, :), CIFAR_DIM);%32*32*3
  patch = patch(r:r+rfSize-1,c:c+rfSize-1,:);%6*6*3
  patches(i,:) = patch(:)';%patches
end

% normalize for contrast，亮度对比度均一化，减去每一行的均值然后除以该行的标准差（其实是标准差加10）
%bsxfun(@rdivide,A,B)表示A中每个元素除以B中对应行（或列）的值。
patches = bsxfun(@rdivide, bsxfun(@minus, patches, mean(patches,2)), sqrt(var(patches,[],2)+10));

% whiten
if (whitening)
  C = cov(patches);%计算patches的协方差矩阵
  M = mean(patches);
  [V,D] = eig(C);
  P = V * diag(sqrt(1./(diag(D) + 0.1))) * V';%P是ZCA Whitening矩阵
  %对数据矩阵白化前，应保证每一维的均值为0
  patches = bsxfun(@minus, patches, M) * P;%注意patches的行列表示的意义不同时，白化矩阵的位置也是不同的。
end

% run K-means
centroids = runkmeans(patches, numCentroids, 50);%对样本数据patches进行聚类，聚类结果保存在centroids中
show_centroids(centroids, rfSize);
drawnow;

% extract training features
if (whitening)
  trainXC = extract_features(trainX, centroids, rfSize, CIFAR_DIM, M,P);%M为均值向量，P为白化矩阵，CIFAR_DIM为patch的维数，rfSize为小patch的大小
else
  trainXC = extract_features(trainX, centroids, rfSize, CIFAR_DIM);
end

% standardize data，保证输入svm分类器中的数据都是标准化过了的
trainXC_mean = mean(trainXC);
trainXC_sd = sqrt(var(trainXC)+0.01);
trainXCs = bsxfun(@rdivide, bsxfun(@minus, trainXC, trainXC_mean), trainXC_sd);
trainXCs = [trainXCs, ones(size(trainXCs,1),1)];%每一个特征后面都添加了一个常量1

% train classifier using SVM
C = 100;
theta = train_svm(trainXCs, trainY, C);

[val,labels] = max(trainXCs*theta, [], 2);
fprintf('Train accuracy %f%%\n', 100 * (1 - sum(labels ~= trainY) / length(trainY)));

%%%%% TESTING %%%%%

%% Load CIFAR test data
fprintf('Loading test data...\n');
f1=load([CIFAR_DIR '/test_batch.mat']);
testX = double(f1.data);
testY = double(f1.labels) + 1;
clear f1;

% compute testing features and standardize
if (whitening)
  testXC = extract_features(testX, centroids, rfSize, CIFAR_DIM, M,P);
else
  testXC = extract_features(testX, centroids, rfSize, CIFAR_DIM);
end
testXCs = bsxfun(@rdivide, bsxfun(@minus, testXC, trainXC_mean), trainXC_sd);
testXCs = [testXCs, ones(size(testXCs,1),1)];

% test and print result
[val,labels] = max(testXCs*theta, [], 2);
fprintf('Test accuracy %f%%\n', 100 * (1 - sum(labels ~= testY) / length(testY)));