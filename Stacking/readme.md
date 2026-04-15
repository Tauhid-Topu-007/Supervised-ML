# Ensemble Learning: Voting & Stacking Regressors + Stacking Classifier

This repository demonstrates three powerful ensemble learning techniques using scikit-learn:
- **Voting Regressor** - Combines multiple regression models
- **Stacking Regressor** - Stacks models with a meta-regressor
- **Stacking Classifier** - Converts regression to binary classification with stacking

## 📋 Overview

The project showcases how ensemble methods can improve predictive performance by combining multiple base models. It includes both regression and classification tasks using synthetic data.

## 🚀 Implementations

### 1. Voting Regressor
Combines multiple regressors using averaging predictions:

```python
voting_reg = VotingRegressor([
    ('lr', LinearRegression()),
    ('rf', RandomForestRegressor()),
    ('svr', SVR())
])
