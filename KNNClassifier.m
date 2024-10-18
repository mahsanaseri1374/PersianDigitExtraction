function [predicted_labels, accuracy, confusion_matrix, timing_per_pattern] = KNNClassifier(train_features, train_labels, test_features, test_labels, k)
    % Ensure that the labels are column vectors
    train_labels = train_labels(:);
    test_labels = test_labels(:);

    % Initialize array to store predicted labels for test data
    predicted_labels = zeros(size(test_labels));

    % Start measuring time for classification
    tic; % Start timing for the entire classification process

    % Loop over each test sample
    for i = 1:size(test_features, 1)
        % Get the i-th test feature
        test_sample = test_features(i, :);

        % Step 1: Calculate the Euclidean distance from the test sample to all training samples
        distances = sqrt(sum((train_features - test_sample).^2, 2));

        % Step 2: Sort distances and get the indices of the k-nearest neighbors
        [~, sorted_indices] = sort(distances, 'ascend');
        k_nearest_indices = sorted_indices(1:k);

        % Step 3: Get the labels of the k-nearest neighbors
        k_nearest_labels = train_labels(k_nearest_indices);

        % Step 4: Assign the most common label (mode) as the predicted label
        predicted_labels(i) = mode(k_nearest_labels);
    end

    % Stop measuring time after classification
    total_time = toc; % Stop the timer

    % Calculate the time per pattern (average time per test sample)
    timing_per_pattern = total_time / size(test_features, 1);  % CPU seconds per pattern

    % Step 5: Calculate accuracy
    accuracy = sum(predicted_labels == test_labels) / numel(test_labels);
    disp(['KNN Classifier Accuracy: ', num2str(accuracy * 100), '%']);

    % Step 6: Generate Confusion Matrix
    confusion_matrix = confusionmat(test_labels, predicted_labels);

    % Display the confusion matrix
    disp('Confusion Matrix:');
    disp(confusion_matrix);

    % Display time information
    disp(['Total Processing Time: ', num2str(total_time * 1000), ' seconds']);
    disp(['Average Time per Pattern: ', num2str(timing_per_pattern * 1000), ' seconds']);
end
