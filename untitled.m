[featuresTransTraining_1] = dataMapping(trainData, netConfig); 
[pred] = softmaxPredict(softmaxModel, featuresTransTraining_1);

acc = mean(testLabels(:) == pred(:));
fprintf('Accuracy: %0.3f%%\n', acc * 100);