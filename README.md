# Optimizing Fake News Detection: A Hybrid Linguistic and Machine Learning Approach

This project aims to enhance the accuracy of fake news detection using a combination of linguistic features and machine learning (ML) techniques. By leveraging the strengths of both approaches, we strive to create a robust and reliable fake news classification system.

## Dataset

The dataset comprises 42,677 news articles, divided into true and fake news categories. Key features include:

Title: The headline of the news article.

Text: The main content of the news article.

Date: The publication date of the news article.

Subject: The topic or category of the news article.

Label: The classification label indicating whether the news is true or fake.

https://www.kaggle.com/datasets/bhavikjikadara/fake-news-detection

This project explores the efficacy of linguistic features in enhancing machine learning models for fake news detection. By examining a diverse set of 88 linguistic features, ranging from surface information to part-of-speech, discursive, and readability indices, this research demonstrates the potential of these features to improve classification accuracy. The study also employs principal component analysis (PCA) for dimensionality reduction and compares the performance of various classifiers, including XGBoost, SVM, and Naive Bayes.

## Eliminating Uninformative Features for Better Machine Learning

The initial analysis revealed 36 features to be unsuitable for our model. These features fell into three
categories:

Missing Values: Some features might have contained a significant number of missing values. This could
happen due to inconsistencies in the data or limitations of the feature extraction process. For instance,
a feature that relies on named entity recognition (identifying people, places, organizations) might have
missing values if the news articles lack proper names.

Low Variability: Certain features might have exhibited very little variation across the dataset. Imagine a
feature that simply counts the number of exclamation marks. If both real and fake news articles rarely
use exclamation points, this feature wouldn't be helpful in distinguishing them.

Little Significance: Statistical tests or feature importance analysis might have shown that some features
had minimal correlation with the target variable (real vs. fake news). These features, despite being
present and varied, might not capture aspects of language that truly differentiate real and fake news.

## Dimensionality Reduction using PCA

Principal Component Analysis (PCA) was employed to reduce the dimensionality of the data while retaining the
most informative features. PCA identifies a set of uncorrelated features, called principal components (PCs), that
explain the maximum variance in the data. The code calculates the explained variance ratio for each PC, indicating
the proportion of variance it captures. By analyzing these ratios, we can determine the number of PCs to retain.
This selection can be based on a chosen threshold for cumulative explained variance (e.g., retaining PCs that
explain 80% of the variance) or by visually inspecting an elbow plot of the explained variance ratio against the PC
number. Focusing on the most informative PCs helps us achieve dimensionality reduction while preserving the
data's key characteristics relevant for fake news classification.

## Machine Learning Model Development

XGBoost, SVM with linear kernel, and SVM with radial kernel are considered some of the best models for
fake news classification due to their ability to effectively handle high-dimensional data and complex
classification tasks. XGBoost, short for Extreme Gradient Boosting, is an ensemble learning method known
for its speed and performance. It works by building multiple decision trees iteratively, each correcting the
errors of the previous ones. This allows XGBoost to capture intricate relationships within the data and make
accurate predictions.

On the other hand, SVM with linear kernel is a powerful linear classifier that works well for datasets with a
clear margin of separation between classes. It constructs a hyperplane that best separates the data points of
different classes, maximizing the margin between the two classes. This makes SVM with linear kernel
particularly suitable for binary classification tasks like fake news detection, where the goal is to distinguish
between two classes (fake and real news) based on linguistic features.
