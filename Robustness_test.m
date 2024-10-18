
clear, close all, clc
load('centered_data.mat');  % Variables: Data (cell array) and labels (double array)

% Rotate all test images by a certain angle (e.g., 15 degrees)
for i = 1:numel(test_data)
    test_data{i} = imrotate(test_data{i}, 15, 'bilinear', 'crop');
end

% Add salt-and-pepper noise to test images
% for i = 1:numel(test_data)
%     test_data{i} = imnoise(test_data{i}, 'salt & pepper', 0.02);
% end

figure;  % Create a new figure window
imshow(test_data{100});  % Display the first image
% title('First Image from Test Data');  % Add a title to the image

[train_horizontal, test_horizontal] = extract_horizontal_features(train_data, test_data);
[train_vertical, test_vertical] = extract_vertical_features(train_data, test_data);
[train_zoning, test_zoning] = extract_zoning_features(train_data, test_data);
[train_gradient, test_gradient] = extract_gradient_features(train_data, test_data);
k = 5;
bandWith = .5;

% horizontal feature
% a = nearestMeanClassifier(train_horizontal, train_labels, test_horizontal, test_labels );
% b = parzenWindowClassifier(train_horizontal, train_labels, test_horizontal, test_labels ,bandWith);
% c = KNNClassifier(train_horizontal, train_labels, test_horizontal, test_labels ,k);
% d = bayesClassifier(train_horizontal, train_labels, test_horizontal, test_labels);


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
a = nearestMeanClassifier(train_gradient, train_labels, test_gradient, test_labels );
b = parzenWindowClassifier(train_gradient, train_labels, test_gradient, test_labels  ,2);
c = KNNClassifier(train_gradient, train_labels, test_gradient, test_labels  ,k);
d = bayesClassifier(train_gradient, train_labels, test_gradient, test_labels );

