# Logistic Regression: Credit Card Default Prediction

## Project Overview

This project applies **logistic regression** to predict whether an individual will default on their credit card debt using the `Default` dataset from the `ISLR2` R package.

The project demonstrates binary classification, model evaluation using a confusion matrix, multicollinearity assessment using Variance Inflation Factors (VIF), and classification performance evaluation using a Receiver Operating Characteristic (ROC) curve.

## Objectives

- Explore and summarize the `Default` dataset.
- Fit a logistic regression model using the complete dataset.
- Evaluate classification performance using a confusion matrix and accuracy.
- Check multicollinearity among predictors using VIF.
- Train a logistic regression model on an 80% training set and evaluate it on a 20% test set.
- Analyze model discrimination using an ROC curve and Area Under the Curve (AUC).

## Dataset

The dataset is available through the `ISLR2` package in R.

```r
library(ISLR2)
data("Default")

View(Default)
str(Default)
summary(Default)
```

The dataset contains information about individuals and whether they defaulted on their credit card debt.

| Variable | Description |
|---|---|
| `default` | Whether the individual defaulted (`Yes` or `No`) |
| `student` | Whether the individual is a student |
| `balance` | Average credit card balance |
| `income` | Individual's income |

The response variable, `default`, is categorical, making logistic regression suitable for this binary classification problem.

## Methodology

### 1. Exploratory Data Analysis

The dataset is examined using `View()`, `str()`, and `summary()` to understand its structure, variable types, and summary statistics.

### 2. Logistic Regression on the Full Dataset

A logistic regression model is fitted using all available predictors.

```r
full_fit <- glm(
  default ~ .,
  data = Default,
  family = binomial
)

summary(full_fit)
```

The model estimates the probability that an individual defaults based on the predictor variables.

### 3. Classification and Confusion Matrix

Predicted probabilities are obtained using:

```r
full.prob <- predict(
  full_fit,
  newdata = Default,
  type = "response"
)
```

A classification threshold of 0.5 is used to assign predicted classes. Predicted probabilities above 0.5 are classified as `Yes`; otherwise, they are classified as `No`.

```r
full.pred <- rep("No", nrow(Default))
full.pred[full.prob > 0.5] <- "Yes"

table(Predicted = full.pred, Actual = Default$default)
```

The confusion matrix compares predicted classifications with actual outcomes.

Classification accuracy is calculated as:

\[
\text{Accuracy} =
\frac{\text{Correct Predictions}}{\text{Total Predictions}}
\]

**Note:** This full-dataset evaluation is an in-sample assessment. A separate test set provides a more reliable estimate of performance on unseen data.

### 4. Multicollinearity Assessment

Variance Inflation Factors (VIF) are calculated using the `car` package.

```r
car::vif(full_fit)
```

VIF measures how strongly a predictor is associated with the other predictors in the model. Large values may indicate multicollinearity, which can make coefficient estimates less stable.

### 5. Training and Testing the Model

The dataset is divided into an 80% training set and a 20% test set.

```r
set.seed(123)

train <- sample(
  seq_len(nrow(Default)),
  size = floor(0.80 * nrow(Default))
)

train.mat <- Default[train, ]
test.mat <- Default[-train, ]

logis.train <- glm(
  default ~ .,
  data = Default,
  family = binomial,
  subset = train
)

summary(logis.train)
```

The training set is used to estimate the model coefficients, while the test set is reserved for evaluating its predictive performance.

### 6. ROC Curve and AUC

Predicted probabilities for the test set are obtained as follows:

```r
pred.prob <- predict(
  logis.train,
  newdata = test.mat,
  type = "response"
)
```

The `pROC` package is used to calculate and plot the ROC curve.

```r
library(pROC)

roc_score <- roc(
  response = test.mat$default,
  predictor = pred.prob,
  levels = c("No", "Yes"),
  direction = "<"
)

roc_score
plot(
  roc_score,
  main = "ROC Curve — Logistic Regression",
  legacy.axes = TRUE,
  xlab = "False Positive Rate",
  ylab = "True Positive Rate"
)
```

The ROC curve plots the true positive rate against the false positive rate across different classification thresholds.

The **Area Under the Curve (AUC
