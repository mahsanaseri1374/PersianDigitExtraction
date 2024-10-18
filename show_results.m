classifiers = {'KNN', 'SVM', 'Bayes', 'RandomForest'};
accuracies = [90, 93, 88, 95];  % Example accuracy for each classifier

figure;
bar(accuracies);
set(gca, 'XTickLabel', classifiers);
ylabel('Accuracy (%)');
title('Accuracy Comparison of Classifiers');

%%
[train_horizontal, test_horizontal] = extract_horizontal_features(train_data, test_data);
[train_vertical, test_vertical] = extract_vertical_features(train_data, test_data);
[train_zoning, test_zoning] = extract_zoning_features(train_data, test_data);
[train_gradient, test_gradient] = extract_gradient_features(train_data, test_data);

[predicted_labels, accuracy, confusion_matrix] = KNNClassifier(train_gradient, train_labels, test_gradient, test_labels ,5);

% Assuming `confusion_matrix` is your confusion matrix for a classifier
% figure;
% heatmap(confusion_matrix, 'Title', 'Confusion Matrix for Classifier X');
% xlabel('Predicted Class');
% ylabel('True Class');

% Visualize confusion matrix using heatmap
figure;
heatmap(confusion_matrix, 'Title', 'Confusion Matrix', ...
    'XLabel', 'Predicted Class', 'YLabel', 'Actual Class');

% Per-class accuracy
num_classes = size(confusion_matrix, 1);
per_class_accuracy = zeros(1, num_classes);

for i = 1:num_classes
    per_class_accuracy(i) = confusion_matrix(i, i) / sum(confusion_matrix(i, :));
    disp(['Class ', num2str(i), ' Accuracy: ', num2str(per_class_accuracy(i) * 100), '%']);
end

% Precision, Recall, and F1-Score for each class
for i = 1:num_classes
    tp = confusion_matrix(i, i);
    fp = sum(confusion_matrix(:, i)) - tp;
    fn = sum(confusion_matrix(i, :)) - tp;
    precision = tp / (tp + fp);
    recall = tp / (tp + fn);
    f1_score = 2 * (precision * recall) / (precision + recall);

    disp(['Class ', num2str(i), ' Precision: ', num2str(precision * 100), '%']);
    disp(['Class ', num2str(i), ' Recall: ', num2str(recall * 100), '%']);
    disp(['Class ', num2str(i), ' F1-Score: ', num2str(f1_score * 100), '%']);
end

%%

accuracies = [90, 93, 88, 95; 85, 91, 86, 92];  % Rows: datasets/folds, Columns: classifiers
figure;
boxplot(accuracies, 'Labels', {'KNN', 'SVM', 'Bayes', 'RandomForest'});
ylabel('Accuracy (%)');
title('Performance Variability Across Classifiers');

%%
% Classifiers and feature data
classifiers = {'KNN', 'Parzen', 'Bayes', 'NearestMean'};
gradient = [94.4 95 18.6 84.4];
horizontal = [72.5 73.1 10.7  54.1];
vertical = [59.6 58.9 11.9 44];
zoning = [86.4 84.4 25.1 72.8];
avg_accuracy = [78.22 77.85 17.27 63.82];

% Grouped data matrix
data = [gradient; horizontal; vertical; zoning; avg_accuracy]';

% Create a grouped bar plot
figure;
bar(data);
xlabel('Classifiers');
ylabel('Percentage (%)');
title('Classifier Performance Comparison');
set(gca, 'XTickLabel', classifiers);
legend({'Gradient', 'Horizontal', 'Vertical', 'Zoning', 'Avg-accuracy'}, 'Location', 'NorthWest');
grid on;

