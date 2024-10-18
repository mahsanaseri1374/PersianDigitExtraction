

function [train_horizontal_features, test_horizontal_features] = extract_horizontal_features(train_data, test_data)
    num_train_images = length(train_data);
    num_test_images = length(test_data);
    image_height = size(train_data{1}, 1); % Height of each image

    % Initialize arrays to store features
    train_horizontal_features = zeros(num_train_images, image_height);
    test_horizontal_features = zeros(num_test_images, image_height);

    % Extract features for all training images
    for i = 1:num_train_images
        image = train_data{i}; % Get the image from the cell
        train_horizontal_features(i, :) = sum(double(image), 2)';  % Transpose to make it a row
    end

    % Extract features for all test images
    for i = 1:num_test_images
        image = test_data{i}; % Get the image from the cell
        test_horizontal_features(i, :) = sum(double(image), 2)';  % Transpose to make it a row
    end
end



% function [train_horizontal_features, test_horizontal_features] = extract_horizontal_features(train_data, test_data)
%     num_train_images = length(train_data);
%     num_test_images = length(test_data);
%     image_height = size(train_data{1}, 1); % Height of each image
% 
%     % Initialize arrays to store features
%     train_horizontal_features = zeros(num_train_images, image_height);
%     test_horizontal_features = zeros(num_test_images, image_height);
% 
%     % Extract features for all training images
%     for i = 1:num_train_images
%         image = double(train_data{i}); % Convert to double
%         train_horizontal_features(i, :) = sum(image, 2)' / size(image, 2);  % Normalize by the width
%     end
% 
%     % Extract features for all test images
%     for i = 1:num_test_images
%         image = double(test_data{i}); % Convert to double
%         test_horizontal_features(i, :) = sum(image, 2)' / size(image, 2);  % Normalize by the width
%     end
% end
