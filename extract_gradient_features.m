function [train_gradient_features, test_gradient_features] = extract_gradient_features(train_data, test_data)
    num_train_images = length(train_data);
    num_test_images = length(test_data);

    % Initialize arrays to store features (200 elements per image based on the paper)
    train_gradient_features = zeros(num_train_images, 200);
    test_gradient_features = zeros(num_test_images, 200);

    % Extract features for all training images
    for i = 1:num_train_images
        image = train_data{i}; % Get the image from the cell
        train_gradient_features(i, :) = sobel_gradient(image); % Compute gradient features
    end

    % Extract features for all test images
    for i = 1:num_test_images
        image = test_data{i}; % Get the image from the cell
        test_gradient_features(i, :) = sobel_gradient(image); % Compute gradient features
    end
end
function gradients = sobel_gradient(image)
    % Convert to grayscale if image is RGB
    if size(image, 3) == 3
        image = rgb2gray(image);
    end
    
    % Sobel operator kernels for x and y gradients
    sobel_x = [-1 0 1; -2 0 2; -1 0 1]; % Sobel kernel for x direction
    sobel_y = [-1 -2 -1; 0 0 0; 1 2 1]; % Sobel kernel for y direction

    % Convolve the image with the Sobel kernels
    gradient_x = conv2(double(image), sobel_x, 'same');
    gradient_y = conv2(double(image), sobel_y, 'same');

    % Calculate gradient strength and direction
    gradient_strength = sqrt(gradient_x.^2 + gradient_y.^2);
    gradient_direction = atan2(gradient_y, gradient_x);

    % Decompose the gradient into the eight Freeman directions
    freeman_directions = [0, pi/4, pi/2, 3*pi/4, pi, -3*pi/4, -pi/2, -pi/4]; % Freeman directions

    % Initialize 8 layers corresponding to Freeman directions
    layers = zeros([size(image), 8]);

    for d = 1:8
        % For each Freeman direction, project the gradient onto the nearest direction
        layer_mask = abs(gradient_direction - freeman_directions(d)) < pi/8; % Nearest direction
        layers(:, :, d) = layer_mask .* gradient_strength;
    end

    % Apply a Gaussian mask to each layer
    gaussian_mask = fspecial('gaussian', [5 5], sqrt(2));
    for d = 1:8
        layers(:, :, d) = conv2(layers(:, :, d), gaussian_mask, 'same');
    end

    % Sample the layers into a 5x5 grid and flatten them into a 200-element feature vector
    sampled_features = zeros(1, 200); % 8 directions * 5x5 grid
    index = 1;
    for d = 1:8
        % Divide the layer into a 5x5 grid and take the average of each grid cell
        layer = layers(:, :, d);
        grid = imresize(layer, [5 5], 'bilinear'); % Resize to 5x5
        sampled_features(index:index+24) = grid(:);
        index = index + 25;
    end

    gradients = sampled_features; % Final 200-element feature vector
end
