% Array to store the accuracy for each combination
clear, close all, clc
load('centered_data.mat');  % Variables: Data (cell array) and labels (double array)

[train_horizontal, test_horizontal] = extract_horizontal_features(train_data, test_data);
[train_vertical, test_vertical] = extract_vertical_features(train_data, test_data);
[train_zoning, test_zoning] = extract_zoning_features(train_data, test_data);
[train_gradient, test_gradient] = extract_gradient_features(train_data, test_data);
k = 5;
bandWith = .5;

accuracies = [];

% Test each feature extraction method with different classifiers
feature_methods = {'horizontal', 'vertical', 'zoning', 'gradient'};
classifiers = {'Nearest Mean', 'Parzen Window', 'KNN', 'Bayes'};

for f = 1:length(feature_methods)
    disp(['Testing ', feature_methods{f}, ' features:']);
    
    % Get training and test features based on the method
    switch feature_methods{f}
        case 'horizontal'
            train_feat = train_horizontal;
            test_feat = test_horizontal;
        case 'vertical'
            train_feat = train_vertical;
            test_feat = test_vertical;
        case 'zoning'
            train_feat = train_zoning;
            test_feat = test_zoning;
        case 'gradient'
            train_feat = train_gradient;
            test_feat = test_gradient;
    end
    
    % Test classifiers on the extracted features
    [~,acc1,~] = nearestMeanClassifier(train_feat, train_labels, test_feat, test_labels);
    [~,acc2,~]  = parzenWindowClassifier(train_feat, train_labels, test_feat, test_labels, bandWith);
     [~,acc3,~]  = KNNClassifier(train_feat, train_labels, test_feat, test_labels, k);
   [~,acc4,~]   = bayesClassifier(train_feat, train_labels, test_feat, test_labels);

    % Store the results
    accuracies = [accuracies; {feature_methods{f}, 'Nearest Mean', acc1}];
    accuracies = [accuracies; {feature_methods{f}, 'Parzen Window', acc2}];
    accuracies = [accuracies; {feature_methods{f}, 'KNN', acc3}];
    accuracies = [accuracies; {feature_methods{f}, 'Bayes', acc4}];
end

% Display the results
disp('Accuracy Results:');
disp(accuracies);
