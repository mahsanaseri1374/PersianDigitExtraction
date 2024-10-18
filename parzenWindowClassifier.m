function [predicted_labels, accuracy, confusion_matrix, timing_per_pattern] = parzenWindowClassifier(train_features, train_labels, test_features, test_labels, bandwidth)
    % Unique class labels and number of classes
    unique_labels = unique(train_labels);
    num_classes = numel(unique_labels);
    num_test_samples = size(test_features, 1);
    predicted_labels = zeros(num_test_samples, 1);

    % Normalize the training and test features
    train_features = normalize(train_features);
    test_features = normalize(test_features);

    % Start measuring time for classification
    tic; % Start timing for the entire classification process

    % Loop through each test sample
    for i = 1:num_test_samples
        % Initialize likelihood for each class
        likelihoods = zeros(num_classes, 1);

        for j = 1:num_classes
            % Get the samples belonging to the current class
            class_samples = train_features(train_labels == unique_labels(j), :);

            % Calculate the density using Parzen window with a Gaussian kernel
            for k = 1:size(class_samples, 1)
                diff = (test_features(i, :) - class_samples(k, :)) / bandwidth;
                likelihood = exp(-0.5 * sum(diff .^ 2)) / ((2 * pi)^(size(train_features, 2)/2) * bandwidth^size(train_features, 2));
                likelihoods(j) = likelihoods(j) + likelihood;
            end

            % Normalize likelihood by the number of samples in the class
            likelihoods(j) = likelihoods(j) / size(class_samples, 1);
        end

        % Predict the class with the highest likelihood
        [~, max_idx] = max(likelihoods);
        predicted_labels(i) = unique_labels(max_idx);  % Use the original label
    end

    % Stop measuring time after classification
    total_time = toc; % Stop the timer

    % Calculate the time per pattern (average time per test sample)
    timing_per_pattern = total_time / num_test_samples;  % CPU seconds per pattern

    % Calculate accuracy
    accuracy = sum(predicted_labels == test_labels) / numel(test_labels) * 100;

    % Confusion matrix
    confusion_matrix = confusionmat(test_labels, predicted_labels);

    % Display results and timing information
    disp(['Parzen Window Classifier Accuracy: ', num2str(accuracy), '%']);
    disp('Confusion Matrix:');
    disp(confusion_matrix);

    disp(['Total Processing Time: ', num2str(total_time * 1000), ' seconds']);
    disp(['Average Time per Pattern: ', num2str(timing_per_pattern * 1000), ' seconds']);
end

function normalized_data = normalize(data)
    % Normalizes the input data by subtracting the mean and dividing by the standard deviation
    mu = mean(data, 1);
    sigma = std(data, 0, 1);
    sigma(sigma == 0) = 1;  % Prevent division by zero in case of constant features
    normalized_data = (data - mu) ./ sigma;
end
