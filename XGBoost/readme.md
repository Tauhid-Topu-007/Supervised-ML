# XGBoost Binary Classification Example

This repository demonstrates a simple yet effective implementation of an **XGBoost classifier** on a synthetic binary classification dataset using scikit-learn.

## 📋 Overview

The project generates a synthetic dataset with 500 samples and 20 features, splits it into training and testing sets, trains an XGBoost classifier, and evaluates its performance using accuracy and classification metrics.

## 🚀 Features

- Synthetic dataset generation with `make_classification`
- Train-test split (70% train, 30% test)
- XGBoost classifier with:
  - 100 estimators (trees)
  - Maximum depth of 3
  - Learning rate of 0.1
  - Log loss evaluation metric
- Performance evaluation with accuracy score and classification report

