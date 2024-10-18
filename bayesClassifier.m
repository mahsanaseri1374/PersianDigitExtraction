% function [predicted_labels, accuracy, confusion_matrix] = bayesClassifier(train_features, train_labels, test_features, test_labels)
%     % تعداد کلاس‌ها
%     classes = unique(train_labels);
%     num_classes = numel(classes);
%     num_samples = size(train_features, 1);
%
%     % محاسبه احتمالات پیشین (Prior Probabilities)
%     prior_probs = zeros(num_classes, 1);
%     for i = 1:num_classes
%         prior_probs(i) = sum(train_labels == classes(i)) / num_samples;
%     end
%
%     % محاسبه میانگین و واریانس برای هر کلاس
%     mean_vals = zeros(num_classes, size(train_features, 2));
%     var_vals = zeros(num_classes, size(train_features, 2));
%
%     for i = 1:num_classes
%         class_samples = train_features(train_labels == classes(i), :);
%         mean_vals(i, :) = mean(class_samples, 1);
%         var_vals(i, :) = var(class_samples, 0, 1);
%     end
%
%     % پیش‌بینی کلاس برای داده‌های تست
%     num_test_samples = size(test_features, 1);
%     predicted_labels = zeros(num_test_samples, 1);
%
%     for i = 1:num_test_samples
%         % احتمال شرطی (Conditional Probabilities)
%         posteriors = zeros(num_classes, 1);
%
%         for j = 1:num_classes
%             % استفاده از توزیع نرمال برای محاسبه احتمال
%             likelihood = exp(-0.5 * ((test_features(i, :) - mean_vals(j, :)).^2) ./ (var_vals(j, :) + eps)); % eps برای جلوگیری از تقسیم بر صفر
%             likelihood = likelihood ./ sqrt(2 * pi * var_vals(j, :) + eps); % محاسبه تابع چگالی نرمال
%
%             % محاسبه احتمال نهایی با توجه به احتمال پیشین
%             posteriors(j) = prior_probs(j) * prod(likelihood);
%         end
%
%         % انتخاب کلاسی که بیشترین احتمال را دارد
%         [~, max_index] = max(posteriors);
%         predicted_labels(i) = classes(max_index);
%     end
%
%     % محاسبه دقت
%     accuracy = sum(predicted_labels == test_labels) / numel(test_labels) * 100;
%
%     % محاسبه ماتریس confusion
%     confusion_matrix = confusionmat(test_labels, predicted_labels);
%
%     disp(['Bayes Classifier Accuracy: ', num2str(accuracy), '%']);
%     disp('Confusion Matrix:');
%     disp(confusion_matrix);
% end
function [predicted_labels, accuracy, confusion_matrix, timing_per_pattern] = bayesClassifier(train_features, train_labels, test_features, test_labels)
    % Get unique classes
    classes = unique(train_labels);
    num_classes = numel(classes);
    num_samples = size(train_features, 1);

    % Calculate prior probabilities
    prior_probs = zeros(num_classes, 1);
    for i = 1:num_classes
        prior_probs(i) = sum(train_labels == classes(i)) / num_samples;
    end

    % Calculate mean and variance for each class
    mean_vals = zeros(num_classes, size(train_features, 2));
    var_vals = zeros(num_classes, size(train_features, 2));

    for i = 1:num_classes
        class_samples = train_features(train_labels == classes(i), :);
        mean_vals(i, :) = mean(class_samples, 1);
        var_vals(i, :) = var(class_samples, 0, 1) + 1e-6;  % Add small constant to avoid division by zero
    end

    % Predict class for test samples
    num_test_samples = size(test_features, 1);
    predicted_labels = zeros(num_test_samples, 1);

    % Start measuring time for classification
    tic;  % Start timing the classification process

    % Use log-likelihood for numerical stability
    for i = 1:num_test_samples
        posteriors = zeros(num_classes, 1);

        for j = 1:num_classes
            % Calculate the log-likelihood using the normal distribution
            log_likelihood = -0.5 * sum(log(2 * pi * var_vals(j, :))) ...
                - 0.5 * sum(((test_features(i, :) - mean_vals(j, :)).^2) ./ var_vals(j, :));

            % Compute the final posterior probability
            posteriors(j) = log(prior_probs(j)) + log_likelihood;
        end

        % Assign the class with the highest posterior probability
        [~, max_index] = max(posteriors);
        predicted_labels(i) = classes(max_index);
    end

    % Stop timing after classification
    total_time = toc;  % Stop the timer

    % Calculate the average time per pattern (test sample)
    timing_per_pattern = total_time / num_test_samples;  % Time per test sample

    % Calculate accuracy
    accuracy = sum(predicted_labels == test_labels) / numel(test_labels) * 100;

    % Generate confusion matrix
    confusion_matrix = confusionmat(test_labels, predicted_labels);

    % Display results and timing information
    disp(['Bayes Classifier Accuracy: ', num2str(accuracy), '%']);
    disp('Confusion Matrix:');
    disp(confusion_matrix);
    disp(['Total Processing Time: ', num2str(total_time * 1000), ' seconds']);
    disp(['Average Time per Pattern: ', num2str(timing_per_pattern * 1000), ' milliseconds']);
end


