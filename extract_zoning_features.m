function [train_zoning_features, test_zoning_features] = extract_zoning_features(train_data, test_data)
    % Parameters
    num_train_images = length(train_data);
    num_test_images = length(test_data);
    zone_size = 5;  % 5x5 zones for zoning feature extraction
    target_size = 20;  % Resize images to 20x20 as per the paper

    % Initialize arrays to store zoning features (5x5 = 25 zones)
    train_zoning_features = zeros(num_train_images, zone_size * zone_size);
    test_zoning_features = zeros(num_test_images, zone_size * zone_size);

    % Extract zoning features for training data
    for i = 1:num_train_images
        image = train_data{i};  % Get the image from the cell
        resized_image = imresize(image, [target_size target_size]);  % Resize to 20x20
        
        % Divide the image into 5x5 zones and compute the mean pixel value for each zone
        train_zoning_features(i, :) = compute_zoning(resized_image, zone_size);
    end

    % Extract zoning features for test data
    for i = 1:num_test_images
        image = test_data{i};  % Get the image from the cell
        resized_image = imresize(image, [target_size target_size]);  % Resize to 20x20
        
        % Divide the image into 5x5 zones and compute the mean pixel value for each zone
        test_zoning_features(i, :) = compute_zoning(resized_image, zone_size);
    end
end

% Helper function to compute zoning feature for one image
function zoning_features = compute_zoning(image, zone_size)
    [image_height, image_width] = size(image);
    zoning_features = zeros(1, zone_size * zone_size);  % Initialize feature vector

    % Compute zone height and width
    zone_height = floor(image_height / zone_size);
    zone_width = floor(image_width / zone_size);

    % Loop through each zone and compute the mean pixel value
    for z_row = 1:zone_size
        for z_col = 1:zone_size
            % Extract the zone (subsection of the image)
            zone = image((z_row-1)*zone_height+1:z_row*zone_height, ...
                         (z_col-1)*zone_width+1:z_col*zone_width);
            
            % Compute mean pixel value for the zone
            zoning_features((z_row-1)*zone_size + z_col) = mean(zone(:));
        end
    end
end
