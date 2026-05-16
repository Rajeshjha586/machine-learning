import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

data = load_breast_cancer()
X, y = data.data, data.target
labels = data.target_names
feature_names = data.feature_names

print(data.DESCR)

print(data.target_names) #['malignant' 'benign']

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Add some noise
# Add Gaussian noise to the data set
np.random.seed(42)  # For reproducibility
noise_factor = 0.5 # Adjust this to control the amount of noise
X_noisy = X_scaled + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=X.shape)

# Load the original and noisy data sets into a DataFrame for comparison and visualization
df = pd.DataFrame(X_scaled, columns=feature_names)
df_noisy = pd.DataFrame(X_noisy, columns=feature_names)

# Display the first few rows of the standardized original and noisy data sets for comparison
print("Original Data (First 5 rows):")
df.head()

# Visualizing the noise content.
plt.figure(figsize=(12, 6))

# Original Feature Distribution (Noise-Free)
plt.subplot(1, 2, 1)
plt.hist(df[feature_names[5]], bins=20, alpha=0.7, color='blue', label='Original')
plt.title('Original Feature Distribution')
plt.xlabel(feature_names[5])
plt.ylabel('Frequency')

# Noisy Feature Distribution
plt.subplot(1, 2, 2)
plt.hist(df_noisy[feature_names[5]], bins=20, alpha=0.7, color='red', label='Noisy')
plt.title('Noisy Feature Distribution')
plt.xlabel(feature_names[5])
plt.ylabel('Frequency')

plt.tight_layout()  # Ensures proper spacing between subplots
plt.show() #The noise-free histogram is skewed to the left and appears to a log-normal distribution, while the noisy histogram is less skewed, tending toward a normal distribution.

# Plots Together
plt.figure(figsize=(12, 6))
plt.plot(df[feature_names[5]], label='Original',lw=3)
plt.plot(df_noisy[feature_names[5]], '--',label='Noisy',)
plt.title('Scaled feature comparison with and without noise')
plt.xlabel(feature_names[5])
plt.legend()
plt.tight_layout()
plt.show()

# Scatterplot
plt.figure(figsize=(12, 6))
plt.scatter(df[feature_names[5]], df_noisy[feature_names[5]],lw=5)
plt.title('Scaled feature comparison with and without noise')
plt.xlabel('Original Feature')
plt.ylabel('Noisy Feature')
plt.tight_layout()
plt.show()

# Split the data, and fit the KNN and SVM models to the noisy training data
# Split the data set into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_noisy, y, test_size=0.3, random_state=42)
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize the models
knn = KNeighborsClassifier(n_neighbors=5)
svm = SVC(kernel='linear', C=1, random_state=42)

# Fit the models to the training data
knn.fit(X_train, y_train)
svm.fit(X_train, y_train)

# Evaluate the models

# Predict on the test set
y_pred_knn = knn.predict(X_test)
y_pred_svm = svm.predict(X_test)

# Print the accuracy scores and classification reports for both models
print(f"KNN Testing Accuracy: {accuracy_score(y_test, y_pred_knn):.3f}")
print(f"SVM Testing Accuracy: {accuracy_score(y_test, y_pred_svm):.3f}")

print("\nKNN Testing Data Classification Report:")
print(classification_report(y_test, y_pred_knn))

print("\nSVM Testing Data Classification Report:")
print(classification_report(y_test, y_pred_svm))

#Plot the confusion matrices

conf_matrix_knn = confusion_matrix(y_test, y_pred_knn)
conf_matrix_svm = confusion_matrix(y_test, y_pred_svm)
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.heatmap(conf_matrix_knn, annot=True, cmap='Blues', fmt='d', ax=axes[0],
            xticklabels=labels, yticklabels=labels)

axes[0].set_title('KNN Testing Confusion Matrix')
axes[0].set_xlabel('Predicted')
axes[0].set_ylabel('Actual')

sns.heatmap(conf_matrix_svm, annot=True, cmap='Blues', fmt='d', ax=axes[1],
            xticklabels=labels, yticklabels=labels)
axes[1].set_title('SVM Testing Confusion Matrix')
axes[1].set_xlabel('Predicted')
axes[1].set_ylabel('Actual')

plt.tight_layout()
plt.show()

# What is the worst kind of prediction error in this context?
# By convention, a positive test for malignancy means a diagnosis of a mass being malignant.
# Thus, a benign prediction is a negative prediction. The worse-case scenario then is a false negative prediction,
# where the test incorrectly predicts that the mass is benign.
# For the KNN model, the number of false negatives is 7, while for the SVM model the count is 2.
# We can say that the SVM model has a higher prediction sensitivity than the KNN model does.

# What can you say to compare the overall performances of the two models
# SVM outperformed KNN in terms of precision, recall, and F1-score
# for both for the individual classes and their overall averages.
# This indicates that SVM is a stronger classifier.
# Although KNN performed quite well with an accuracy of 94%,
# VM has better ability to correctly classify both malignant and beinign cases, with fewer errors.
# Given that the goal would be to choose the model with better generalization and fewer false negatives,
# SVM is certainly the preferred classifier.

# Obtain the prediction results using the training data.
y_pred_train_knn = knn.predict(X_train)
y_pred_train_svm = svm.predict(X_train)

# Evaluate the models on the training data
print(f"KNN Training Accuracy: {accuracy_score(y_train, y_pred_train_knn):.3f}")
print(f"SVM Training Accuracy: {accuracy_score(y_train, y_pred_train_svm):.3f}")

print("\nKNN Training Classification Report:")
print(classification_report(y_train, y_pred_train_knn))

print("\nSVM Training Classification Report:")
print(classification_report(y_train, y_pred_train_svm))

# Plot the confusion matrices for the training data
# Enter your code here
# Plot the confusion matrices
conf_matrix_knn = confusion_matrix(y_train, y_pred_train_knn)
conf_matrix_svm = confusion_matrix(y_train, y_pred_train_svm)
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.heatmap(conf_matrix_knn, annot=True, cmap='Blues', fmt='d', ax=axes[0],
            xticklabels=labels, yticklabels=labels)

axes[0].set_title('KNN Training Confusion Matrix')
axes[0].set_xlabel('Predicted')
axes[0].set_ylabel('Actual')

sns.heatmap(conf_matrix_svm, annot=True, cmap='Blues', fmt='d', ax=axes[1],
            xticklabels=labels, yticklabels=labels)
axes[1].set_title('SVM Training Confusion Matrix')
axes[1].set_xlabel('Predicted')
axes[1].set_ylabel('Actual')

plt.tight_layout()
plt.show()

sns.heatmap(..., annot=True, cmap='Blues', fmt='d', ax=axes[1],
            xticklabels=labels, yticklabels=labels)
axes[1].set_title('SVM Training Confusion Matrix')
axes[1].set_xlabel('Predicted')
axes[1].set_ylabel('Actual')

plt.tight_layout()
plt.show()

# Comparing training and testing accuracies for both models
# Ideally, a well-performing model should have similar accuracy
# on both the training and testing datasets.

# It is unusual for test accuracy to be higher than training accuracy.
# This may happen due to random chance or data leakage.
# For example, in this case, the entire dataset was normalized before
# splitting, instead of fitting StandardScaler only on training data
# and then applying the same transformation to both train and test sets.
# This and similar pitfalls will be covered in later labs.

# If training accuracy is much higher than testing accuracy,
# the model may be memorizing patterns from training data that
# do not generalize well to unseen data. This is called overfitting.

# For the SVM model, training and testing accuracies are nearly equal
# at around 97%, which is ideal and suggests the model is not overfitting.

# For the KNN model, training accuracy is about 2% higher than test accuracy,
# indicating a small amount of overfitting may be present.

# In summary, the SVM model is more reliable and achieves higher accuracy
# than the KNN model in this comparison.

# Note: The goal here is not hyperparameter tuning, but simply comparing
# model performance using a fixed set of hyperparameters.

# KNN	Train	95.5%
# KNN	Test	93.6%
# SVM	Train	97.2%
# SVM	Test	97.1%