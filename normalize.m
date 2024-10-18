
function normalized_data = normalize(data)
% Normalizes the input data by subtracting the mean and dividing by the standard deviation
mu = mean(data, 1);
sigma = std(data, 0, 1);
sigma(sigma == 0) = 1;  % Prevent division by zero in case of constant features
normalized_data = (data - mu) ./ sigma;
end
