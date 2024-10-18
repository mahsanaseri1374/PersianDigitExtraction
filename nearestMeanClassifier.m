function [predicted_labels, accuracy, confusion_matrix, timing_per_pattern] = nearestMeanClassifier(train_features, train_labels, test_features, test_labels)
    % Ensure that the labels are column vectors
    train_labels = train_labels(:);
    test_labels = test_labels(:);

    % Step 1: Find unique classes in training data
    unique_classes = unique(train_labels);

    % Step 2: Calculate mean feature vector for each class
    class_means = zeros(length(unique_classes), size(train_features, 2));
    for i = 1:length(unique_classes)
        class_samples = train_features(train_labels == unique_classes(i), :);
        class_means(i, :) = mean(class_samples, 1);
    end

    % Step 3: Initialize array to store predicted labels for test data
    predicted_labels = zeros(size(test_labels));

    % Start measuring time for classification
    tic; % Start timing for the entire classification process

    % Step 4: Classify each test sample by finding the nearest class mean
    for i = 1:size(test_features, 1)
        test_sample = test_features(i, :);

        % Calculate the Euclidean distance to each class mean
        distances = sqrt(sum((class_means - test_sample).^2, 2));

        % Find the class with the minimum distance
        [~, min_index] = min(distances);
        predicted_labels(i) = unique_classes(min_index);
    end

    % Stop measuring time after classification
    total_time = toc; % Stop the timer

    % Calculate the time per pattern (average time per test sample)
    timing_per_pattern = total_time / size(test_features, 1);  % CPU seconds per pattern

    % Step 5: Calculate accuracy
    accuracy = sum(predicted_labels == test_labels) / numel(test_labels);
    disp(['Nearest Mean Classifier Accuracy: ', num2str(accuracy * 100), '%']);

    % Step 6: Generate Confusion Matrix
    confusion_matrix = confusionmat(test_labels, predicted_labels);

    % Display the confusion matrix
    disp('Confusion Matrix:');
    disp(confusion_matrix);

    % Display time information
    disp(['Total Processing Time: ', num2str(total_time * 1000), ' seconds']);
    disp(['Average Time per Pattern: ', num2str(timing_per_pattern * 1000), ' seconds']);
end
