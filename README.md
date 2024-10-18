# Persian Digit Recognition Using Feature Extraction and Classification Techniques

## Overview

This repository contains the code and resources for the project **" Feature Extraction and Classification of Persian Handwritten Digits. "**. The focus of the project is to explore various feature extraction methods and classifiers to achieve accurate digit recognition, even under conditions of noise and image rotation.

## Table of Contents

- [Introduction](#introduction)
- [Feature Extraction](#feature-extraction)
- [Classifiers](#classifiers)
- [Robustness Testing](#robustness-testing)
- [Results](#results)
- [Conclusion](#conclusion)
- [How to Use](#how-to-use)
- [Acknowledgments](#acknowledgments)

## Introduction

In this project, we explore the effectiveness of several **feature extraction** techniques combined with various **classifiers** for digit recognition. The goal is to achieve high accuracy in classifying digits, while also ensuring robustness against **noise** and **rotations**.

## Feature Extraction

We implemented four key feature extraction methods:
1. **Zoning** - Dividing the image into zones and analyzing pixel distributions.
2. **Gradient Features** - Extracting directional gradients to capture edge information.
3. **Horizontal & Vertical Histograms** - Summarizing the pixel intensity distribution along the axes.

## Classifiers

We evaluated the following classifiers:
1. **k-Nearest Neighbors (k-NN)** - Known for its simplicity and effectiveness in high-dimensional spaces.
2. **Parzen Window (with Gaussian Kernel)** - Non-parametric density estimation method.
3. **Bayesian Classifier** - Utilizing probabilistic models based on prior and likelihood calculations.
4. **Nearest Mean Classifier** - Assigns classes based on the proximity to the class mean.

## Robustness Testing

To test the robustness of our system, we introduced:
- **Salt-and-pepper noise** affecting 2% of the pixels.
- **Rotations** of 15 degrees.

The k-NN classifier demonstrated high resilience to image rotation, while the Parzen Window classifier (after feature normalization) showed notable robustness to noise.

## Results

| Feature Extraction | k-NN Accuracy | Parzen Accuracy | Bayesian Accuracy | Nearest Mean Accuracy |
|--------------------|---------------|-----------------|-------------------|-----------------------|
| Zoning             | **98.5%**     | 97.8%           | 92.4%             | 95.3%                 |
| Gradient           | **99.1%**     | **98.9%**       | 93.5%             | 96.7%                 |
| Horizontal Hist.   | 95.4%         | 93.2%           | 89.1%             | 91.6%                 |
| Vertical Hist.     | 94.7%         | 92.1%           | 87.8%             | 90.4%                 |

Based on our experiments:
- The **Gradient** and **Zoning** features provided the highest accuracy across all classifiers.
- The **k-NN** classifier consistently performed the best, especially for larger datasets.
- The **Parzen Window** classifier, after normalization, showed excellent resilience to noise.

## Conclusion

In conclusion, this study highlights the importance of both **feature extraction** and **classifier selection** in creating an effective digit recognition system. **k-NN** proved to be the most robust classifier, particularly when dealing with noisy and rotated data. Additionally, the **Gradient** and **Zoning** features were the most effective in preserving classification accuracy across all tests.

## How to Use

### Prerequisites

- MATLAB or Octave environment
- Clone the repository:
    ```bash
    git clone https://github.com/mahsanaseri1374/PersianDigitExtraction.git
    ```
- Download and load the dataset in the `centered_data.mat` file.

### Running the Code

- Preprocess the data by adding noise or rotating the images:
    ```matlab
    for i = 1:numel(test_data)
        test_data{i} = imnoise(test_data{i}, 'salt & pepper', 0.02);  % Add noise
    end
    ```

- Extract features:
    ```matlab
    [train_zoning, test_zoning] = extract_zoning_features(train_data, test_data);
    ```

- Test different classifiers:
    ```matlab
    k = 5;
    [accuracy, confusion_matrix] = KNNClassifier(train_zoning, train_labels, test_zoning, test_labels, k);
    ```

### Results

After running the experiments, you can compare the accuracy of different classifiers and analyze the impact of noise and rotations on performance.

## Acknowledgments

This work was part of the research at Sharif University of Technology, School of Electrical Engineering. Special thanks to the professors and colleagues for their support throughout the project.

---

_Developed by: [Mahsa Naseri](https://github.com/mahsanaseri1374)_

