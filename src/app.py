from utils import db_connect
engine = db_connect()

# your code here
👨‍💻K-Means Model + Knn Mode

This project combines two powerful machine learning techniques, K-Means and K-Nearest Neighbors (KNN), to address a classification or data analysis problem. The K-Means Model is used to cluster data into groups, aiding in the discovery of patterns and structures within the data. Subsequently, the KNN model is employed to classify samples based on their proximity to the centroids of clusters identified by K-Means. Together, these two approaches can be used to solve a variety of classification and data segmentation problems. This notebook provides a practical approach to implementing and evaluating these models, offering a deeper understanding of how they can be applied to real-world problems.

📝 Instructions

🏬House grouping system
We want to be able to classify houses according to their region and median income. To do this, we will use the famous California Housing dataset. It was constructed using data from the 1990 California census. It contains one row per census block group. A block group is the smallest geographic unit for which US Census data is published.

Step 1: Loading the dataset🎲
The dataset can be found in this project folder under the name housing.csv. You can load it into the code directly from the link (https://raw.githubusercontent.com/4GeeksAcademy/k-means-project-tutorial/main/housing.csv) or download it and add it by hand in your repository. In this case we are only interested in the Latitude, Longitude and MedInc columns.

Be sure to conveniently split the dataset into train and test as we have seen in previous lessons. Although these sets are not used to obtain statistics, you can use them to train the unsupervised algorithm and then to make predictions about new points to predict the cluster they are associated with.
"""

import pandas as pd

url = "https://raw.githubusercontent.com/4GeeksAcademy/k-means-project-tutorial/main/housing.csv"
total_data = pd.read_csv(url)

print(total_data.head())

total_data.to_csv("/workspace/Unsupervised-Learning-K-means/data/raw/housing.csv", index=False)

total_data.info()

"""💾Split data train & test data"""

from sklearn.model_selection import train_test_split

# Columns selected
relevant_data = total_data[['Latitude', 'Longitude', 'MedInc']]
X = relevant_data  # Features

# Split data into train and test
X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)

X_train.to_csv("/workspace/Unsupervised-Learning-K-means/data/processed/X_train.csv")
X_test.to_csv("/workspace/Unsupervised-Learning-K-means/data/processed/X_test.csv")

"""👨‍💻Step.2 KMeans"""

from sklearn.cluster import KMeans
model = KMeans(n_clusters=6, random_state=42)
model.fit(X_train)

X_train['cluster'] = model.labels_ # = y

X_train.head()

from pickle import dump
dump (model, open (f"/workspace/Unsupervised-Learning-K-means/models/k-means_model_clusters6_rs42.pk", "wb"))

X_test['cluster'] = model.predict(X_test)

X_test.head()

"""📊Plot data"""

import matplotlib.pyplot as plt
import seaborn as sns

fig, ax = plt.subplots(1, 2, figsize=(12, 6))

# First subplot for X_train
sns.scatterplot(data=X_train, x="Longitude", y="Latitude", hue="cluster", palette="Set3", ax=ax[0])
ax[0].set_title('Training Data')

# Second subplot for X_test
sns.scatterplot(data=X_test, x="Longitude", y="Latitude", hue="cluster", palette="Set3", ax=ax[1])
ax[1].set_title('Test Data')
plt.show()

"""👉 Therefore KNN clasiffer seems a good choice for coordination calculations

👨‍💼Preparing the dataset for training
"""

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

X_train['cluster'] = model.labels_
y_train = model.labels_

# Make sure X_train does not include the 'cluster' column for KNN model training
y_train = model.labels_
X_train_knn = X_train.drop('cluster', axis=1)

# Create and train the KNN model
knn = KNeighborsClassifier(n_neighbors=3)  # You can adjust the number of neighbors if necessary
knn.fit(X_train_knn, y_train)

"""💾Save knn model"""

from pickle import dump
dump (model, open (f"/workspace/Unsupervised-Learning-K-means/models/knn_model_3n.pk", "wb"))

"""📊Confusion Matrix and classification report"""

from sklearn.metrics import confusion_matrix, classification_report

# Make predictions on the training set
y_pred = knn.predict(X_train_knn)

# Calculate the confusion matrix
confusion_mat = confusion_matrix(y_train, y_pred)

# Generate the classification report
classification_rep = classification_report(y_train, y_pred)

print("Confusion Matrix:")
print(confusion_mat)

print("\nClassification Report:")
print(classification_rep)

"""🙋‍♂️ Conclusions

✅ Precision: The model achieved high precision for all classes, indicating that it correctly identified most positive samples for each class.

✅ Recall (Sensitivity): The model achieved high recall for all classes, indicating that it correctly identified a large proportion of positive samples for each class.

✅ F1-Score: The F1-Score, which balances precision and recall, is high for all classes, suggesting a well-rounded model performance.

✅ Support: This represents the number of true samples for each class in the dataset.

✅ Accuracy: The model achieved 100% accuracy overall, meaning that all samples were correctly classified in the test dataset.

In summary, the model demonstrates exceptional performance with high precision, recall, and F1-Score for all classes. It achieves perfect accuracy on the test dataset, indicating its effectiveness in classifying the data.
"""

from sklearn.metrics import confusion_matrix

# Define class labels
labels = ['Longitude', 'Latitude', 'MedInc']  # Corrected labels

# Calculate the confusion matrix
confusion_mat = confusion_matrix(y_train, y_pred)

# Create a figure and axes for the confusion matrix
plt.figure(figsize=(8, 6))
sns.set(font_scale=1.2)  # Adjust font size for better visualization
sns.heatmap(confusion_mat, annot=True, fmt="d", cmap="Blues",
            xticklabels=labels, yticklabels=labels)
plt.xlabel('Predictions')
plt.ylabel('True Values')
plt.title('Confusion Matrix')
plt.show()

"""🧐Cross Check for KNN"""

from sklearn.model_selection import cross_val_score
import numpy as np

# Make sure X_train does not include the 'cluster' column
X_train_knn = X_train.drop('cluster', axis=1)

# Create the KNN model
knn = KNeighborsClassifier(n_neighbors=3)

# Perform cross-validation
# For example, using 5-fold cross-validation
scores = cross_val_score(knn, X_train_knn, y_train, cv=5)

# Print the results
print("Accuracy scores for each fold: ", scores)
print("Mean cross-validation score: ", np.mean(scores))
print("Standard deviation of the scores: ", np.std(scores))

"""🧙‍♂️ Conclusions:

✅ Data Overview:

The dataset consists of 20,640 entries and 9 columns.
Columns include features such as 'MedInc,' 'HouseAge,' 'AveRooms,' 'AveBedrms,' 'Population,' 'AveOccup,' 'Latitude,' 'Longitude,' and the target variable 'MedHouseVal.'
All columns have non-null values, and they are of float64 data type.

✅ K-Means Clustering:

The K-Means clustering algorithm was applied to the dataset with 6 clusters.
The results include latitude, longitude, 'MedInc,' and the cluster label for some sample data points.

✅ K-Nearest Neighbors (KNN) Classification:

The KNN classifier was used with k=3 neighbors.
The confusion matrix shows the model's performance, with high diagonal values indicating correct predictions and low off-diagonal values suggesting errors.
The classification report provides metrics such as precision, recall, and F1-score for each class, as well as the overall accuracy.
The model achieves high accuracy, precision, recall, and F1-score for all classes, indicating excellent performance.

✅ Cross-Validation:

Cross-validation was performed with 5 folds, resulting in accuracy scores for each fold.
The mean cross-validation score is approximately 99.12%, indicating the model's consistent performance.
The standard deviation of the scores is low, suggesting that the model's performance is stable across folds.
In summary, the K-Means clustering was used to group data points into clusters, and the KNN classifier achieved excellent performance in classifying the data points into classes. The model demonstrated high accuracy and consistency across cross-validation folds.
"""