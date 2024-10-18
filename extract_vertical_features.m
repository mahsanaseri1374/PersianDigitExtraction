function [train_vertical_features, test_vertical_features] = extract_vertical_features(train_data, test_data)
    num_train_images = length(train_data);
    num_test_images = length(test_data);
    image_width = size(train_data{1}, 2); % Width of each image

    % Initialize arrays to store features
    train_vertical_features = zeros(num_train_images, image_width);
    test_vertical_features = zeros(num_test_images, image_width);

    % Extract features for all training images
    for i = 1:num_train_images
        image = train_data{i}; % Get the image from the cell
        train_vertical_features(i, :) = sum(double(image), 1); 
    end

    % Extract features for all test images
    for i = 1:num_test_images
        image = test_data{i}; % Get the image from the cell
        test_vertical_features(i, :) = sum(double(image), 1); 
    end
end


% function [train_vertical_features, test_vertical_features] = extract_vertical_features(train_data, test_data)
%     num_train_images = length(train_data);
%     num_test_images = length(test_data);
%     image_width = size(train_data{1}, 2); % Width of each image
% 
%     % Initialize arrays to store features
%     train_vertical_features = zeros(num_train_images, image_width);
%     test_vertical_features = zeros(num_test_images, image_width);
% 
%     % Extract features for all training images
%     for i = 1:num_train_images
%         image = double(train_data{i}); % Convert to double
%         train_vertical_features(i, :) = sum(image, 1) / size(image, 1);  % Normalize by the height
%     end
% 
%     % Extract features for all test images
%     for i = 1:num_test_images
%         image = double(test_data{i}); % Convert to double
%         test_vertical_features(i, :) = sum(image, 1) / size(image, 1);  % Normalize by the height
%     end
% end

