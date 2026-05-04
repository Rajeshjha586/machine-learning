import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv('https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/teleCust1000t.csv')
# print(df.head())

# print(df['custcat'].value_counts())

correlation_matrix = df.corr()

# plt.figure(figsize=(10, 8))
# sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
# plt.show()

correlation_values = abs(df.corr()['custcat'].drop('custcat')).sort_values(ascending=False)
# print(correlation_values)

X = df.drop('custcat',axis=1)
y = df['custcat']

X_norm = StandardScaler().fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_norm, y, test_size=0.2, random_state=4)

k = 3
#Train Model and Predict
knn_classifier = KNeighborsClassifier(n_neighbors=k)
knn_model = knn_classifier.fit(X_train,y_train)

yhat = knn_model.predict(X_test)

print("Test set Accuracy: ", accuracy_score(y_test, yhat))

k = 6
knn_model_6 = KNeighborsClassifier(n_neighbors = k).fit(X_train,y_train)
yhat6 = knn_model_6.predict(X_test)
print("Test set Accuracy: ", accuracy_score(y_test, yhat6))

Ks = 100
acc = np.zeros((Ks))
std_acc = np.zeros((Ks))
for n in range(1,Ks+1):
    #Train Model and Predict
    knn_model_n = KNeighborsClassifier(n_neighbors = n).fit(X_train,y_train)
    yhat = knn_model_n.predict(X_test)
    acc[n-1] = accuracy_score(y_test, yhat)
    std_acc[n-1] = np.std(yhat==y_test)/np.sqrt(yhat.shape[0])

plt.plot(range(1,Ks+1),acc,'g')
plt.fill_between(range(1,Ks+1),acc - 1 * std_acc,acc + 1 * std_acc, alpha=0.10)
plt.legend(('Accuracy value', 'Standard Deviation'))
plt.ylabel('Model Accuracy')
plt.xlabel('Number of Neighbors (K)')
plt.tight_layout()
plt.show()

# Plot the variation of the accuracy score for the training set for 100 value of Ks.

# Ks =100
# acc = np.zeros((Ks-1))
# std_acc = np.zeros((Ks-1))
# for n in range(1,Ks):
#     #Train Model and Predict
#     knn_model_n = KNeighborsClassifier(n_neighbors = n).fit(X_train,y_train)
#     yhat = knn_model_n.predict(X_train)
#     acc[n-1] = accuracy_score(y_train, yhat)
#     std_acc[n-1] = np.std(yhat==y_train)/np.sqrt(yhat.shape[0])
#
# plt.plot(range(1,Ks),acc,'g')
# plt.fill_between(range(1,Ks),acc - 1 * std_acc, acc + 1 * std_acc, alpha=0.10)
# plt.legend(('Accuracy value', 'Standard Deviation'))
# plt.ylabel('Model Accuracy')
# plt.xlabel('Number of Neighbors (K)')
# plt.tight_layout()
# plt.show()

print( "The best accuracy was with", acc.max(), "with k =", acc.argmax()+1)


# Q1 Can you justify why the model performance on training data is deteriorating with increase in the value of k?
# When k is small (e.g., k=1), the model is highly sensitive to the individual points in the dataset. The prediction for each point is based on its closest neighbor, which can lead to highly specific and flexible boundaries. This leads to overfitting on the training data, meaning the model will perform very well on the training set, potentially achieving 100% accuracy. However, it may generalize poorly to unseen data. When k is large, the model starts to take into account more neighbors when making predictions. This has two main consequences:
#
# Smoothing of the Decision Boundary: The decision boundary becomes smoother, which means the model is less sensitive to the noise or fluctuations in the training data.
# Less Specific Predictions: With a larger k, the model considers more neighbors and therefore makes more generalized predictions, which can lead to fewer instances being classified perfectly.
# As a result, the model starts to become less flexible, and its ability to memorize the training data (which can lead to perfect accuracy with small k) is reduced.


# Q2 We can see that even the with the optimum values, the KNN model is not performing that well on the given data set. Can you think of the possible reasons for this?
# The weak performance on the model can be due to multiple reasons.
# 1. The KNN model relies entirely on the raw feature space at inference time.
# If the features do no provide clear boundaries between classes,
# KNN model cannot compensate through optimization or feature transformation.
# 2. For a high number of weakly correlated features, the number of dimensions increases,
# the distance between points tend to become more uniform, reducing the discriminative power of KNN.
# 3. The algorithm treats all features equally when computing distances.
#
# Hence, weakly correalted features can introduce noise or irrelevant variations in the feature space making it harder for KNN to find meaningful neighbours.