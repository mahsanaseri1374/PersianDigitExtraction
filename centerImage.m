function centered_img = centerImage(image, fixed_size)
    % Convert the image to binary (thresholding)
    binary_img = image > 0;
    
    % Find the bounding box of the digit in the image
    [rows, cols] = find(binary_img);
    top_row = min(rows);
    bottom_row = max(rows);
    left_col = min(cols);
    right_col = max(cols);
    
    % Crop the image to remove empty space around the digit
    cropped_img = image(top_row:bottom_row, left_col:right_col);
    
    % Get the size of the cropped image
    [cropped_height, cropped_width] = size(cropped_img);
    
    % Create an empty image with the fixed size
    centered_img = zeros(fixed_size, fixed_size);
    
    % Calculate the position to place the cropped image in the center
    row_offset = floor((fixed_size - cropped_height) / 2);
    col_offset = floor((fixed_size - cropped_width) / 2);
    
    % Place the cropped image in the center of the frame
    centered_img(row_offset + (1:cropped_height), col_offset + (1:cropped_width)) = cropped_img;
end