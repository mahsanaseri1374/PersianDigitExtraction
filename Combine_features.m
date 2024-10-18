clear, close all, clc

load('centered_data.mat');  % Variables: Data (cell array) and labels (double array)

% Assuming you already have the extracted features as follows:

[train_horizontal, test_horizontal] = extract_horizontal_features(train_data, test_data);
[train_vertical, test_vertical] = extract_vertical_features(train_data, test_data);
[train_zoning, test_zoning] = extract_zoning_features(train_data, test_data);
[train_gradient, test_gradient] = extract_gradient_features(train_data, test_data);

% Combine all features into a single feature matrix for training
% train_features = [train_horizontal, train_vertical, train_zoning, train_gradient];
train_features = [train_horizontal, train_vertical, train_zoning, train_gradient];

% Combine all features into a single feature matrix for testing
% test_features = [test_horizontal, test_vertical, test_zoning, test_gradient];
test_features = [test_horizontal, test_vertical, test_zoning, test_gradient];

% Now you can use `train_features` and `test_features` for your classifier
a = nearestMeanClassifier(train_features, train_labels, test_features, test_labels );
b = parzenWindowClassifier(train_features, train_labels, test_features, test_labels ,.5 );
c = KNNClassifier(train_features, train_labels, test_features, test_labels ,5);
d = bayesClassifier(train_features, train_labels, test_features, test_labels );
