clear, close all, clc

% Load the data (assuming it's stored in 'data.mat' file)
load('Data_hoda_full.mat');  % Variables: Data (cell array) and labels (double array)
%% part 1 dividing
% Split data into training and test sets
train_data = Data(1:3000);    % First 3,000 images for training
train_labels = labels(1:3000);  % Corresponding labels for training

test_data = Data(3001:4000);   % Next 1,000 images for testing
test_labels = labels(3001:4000);  % Corresponding labels for testing

%% part 2 centering

% Step 1: Determine the maximum size (height and width) of all images in the dataset
max_height = 0;
max_width = 0;

% Loop through the training data to find the maximum height and width
for i = 1:numel(train_data)
    [h, w] = size(train_data{i});
    if h > max_height
        max_height = h;
    end
    if w > max_width
        max_width = w;
    end
end

% Do the same for the test data
for i = 1:numel(test_data)
    [h, w] = size(test_data{i});
    if h > max_height
        max_height = h;
    end
    if w > max_width
        max_width = w;
    end
end

% Set the fixed size to the maximum height and width found
fixed_size = max(max_height, max_width);

% Step 2: Centering the images in the calculated fixed size


% Apply centering to all training images with the dynamic fixed size
for i = 1:numel(train_data)
    train_data{i} = centerImage(train_data{i}, fixed_size);
end

% Apply centering to all test images with the dynamic fixed size
for i = 1:numel(test_data)
    test_data{i} = centerImage(test_data{i}, fixed_size);
end
save('centered_data.mat', 'train_data','train_labels', 'test_data','test_labels');
% save('centered_data(500).mat', 'train_data','train_labels', 'test_data','test_labels');
% save('centered_data(1500).mat', 'train_data','train_labels', 'test_data','test_labels');
% save('centered_data(2500).mat', 'train_data','train_labels', 'test_data','test_labels');


