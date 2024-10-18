clear, close all, clc
load('centered_data.mat');  % Variables: Data (cell array) and labels (double array)

[train_horizontal, test_horizontal] = extract_horizontal_features(train_data, test_data);
[train_vertical, test_vertical] = extract_vertical_features(train_data, test_data);
[train_zoning, test_zoning] = extract_zoning_features(train_data, test_data);
[train_gradient, test_gradient] = extract_gradient_features(train_data, test_data);
k = 5;
bandWith = .5;
% 
% train_horizontal = normalize(train_horizontal);
% test_horizontal = normalize(test_horizontal);
% 
% train_vertical = normalize(train_vertical);
% test_vertical = normalize(test_vertical);
% 
% train_zoning = normalize(train_zoning);
% test_zoning = normalize(test_zoning);
% 
% train_gradient = normalize(train_gradient);
% test_gradient = normalize(test_gradient);



% horizontal feature
a = nearestMeanClassifier(train_horizontal, train_labels, test_horizontal, test_labels );
b = parzenWindowClassifier(train_horizontal, train_labels, test_horizontal, test_labels ,bandWith);
c = KNNClassifier(train_horizontal, train_labels, test_horizontal, test_labels ,k);
d = bayesClassifier(train_horizontal, train_labels, test_horizontal, test_labels);


% % vertical feature
% a = nearestMeanClassifier(train_vertical, train_labels, test_vertical, test_labels );
% b = parzenWindowClassifier(train_vertical, train_labels, test_vertical, test_labels ,bandWith);
% c = KNNClassifier(train_vertical, train_labels, test_vertical, test_labels ,k);
% d = bayesClassifier(train_vertical, train_labels, test_vertical, test_labels);


% % zoning feature
% a = nearestMeanClassifier(train_zoning, train_labels, test_zoning, test_labels );
% b = parzenWindowClassifier(train_zoning, train_labels, test_zoning, test_labels  ,bandWith);
% c = KNNClassifier(train_zoning, train_labels, test_zoning, test_labels  ,k);
% d = bayesClassifier(train_zoning, train_labels, test_zoning, test_labels );


% % gradient feature
% a = nearestMeanClassifier(train_gradient, train_labels, test_gradient, test_labels );
% b = parzenWindowClassifier(train_gradient, train_labels, test_gradient, test_labels  ,bandWith);
% c = KNNClassifier(train_gradient, train_labels, test_gradient, test_labels  ,k);
% d = bayesClassifier(train_gradient, train_labels, test_gradient, test_labels );





% best parzen Window
% bandwidths = [0.01, 0.1, 0.5, 1, 2, 5]; % Example bandwidth values
% best_accuracy = 0;
% best_bandwidth = 0;
%
% for b = bandwidths
%     [~, accuracy, ~] = parzenWindowClassifier(train_features, train_labels, test_features, test_labels , b);
%     if accuracy > best_accuracy
%         best_accuracy = accuracy;
%         best_bandwidth = b;
%     end
%     disp(['Bandwidth: ', num2str(b), ' - Accuracy: ', num2str(accuracy), '%']);
% end
%
% disp(['Best Bandwidth: ', num2str(best_bandwidth)]);
% disp(['Best Accuracy: ', num2str(best_accuracy)]);


