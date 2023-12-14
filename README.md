# 👨‍💻 K-Means Model + KNN Model

![image](https://github.com/Munchkinland/Unsupervised-Learning-K-means/assets/92251234/5d6efa89-b98b-4e67-b9b8-40614cee2c66)

This project combines two powerful machine learning techniques, K-Means and K-Nearest Neighbors (KNN), to address a classification or data analysis problem. The K-Means Model is used to cluster data into groups, aiding in the discovery of patterns and structures within the data. Subsequently, the KNN model is employed to classify samples based on their proximity to the centroids of clusters identified by K-Means. Together, these two approaches can be used to solve a variety of classification and data segmentation problems. This notebook provides a practical approach to implementing and evaluating these models, offering a deeper understanding of how they can be applied to real-world problems.

📝 Instructions

🏬 House grouping system:
We want to be able to classify houses according to their region and median income. To do this, we will use the famous California Housing dataset. It was constructed using data from the 1990 California census. It contains one row per census block group. A block group is the smallest geographic unit for which US Census data is published.

**Step 1: Loading the dataset** 🎲
The dataset can be found in this project folder under the name `housing.csv`. You can load it into the code directly from the link (`https://raw.githubusercontent.com/4GeeksAcademy/k-means-project-tutorial/main/housing.csv`) or download it and add it by hand in your repository. In this case, we are only interested in the `Latitude`, `Longitude`, and `MedInc` columns.

Be sure to conveniently split the dataset into train and test as we have seen in previous lessons. Although these sets are not used to obtain statistics, you can use them to train the unsupervised algorithm and then to make predictions about new points to predict the cluster they are associated with.

**Step 2: K-Means** 👨‍💻

We use the K-Means clustering algorithm with 6 clusters to classify houses. The results include latitude, longitude, `MedInc`, and the cluster label.

**Step 3: K-Nearest Neighbors (KNN) Classification** 👨‍💼

We apply K-Nearest Neighbors (KNN) classification to the clustered data. The KNN classifier is trained to predict the cluster label for each house based on its coordinates and median income.

📊 Plot Data

We visualize the data on two plots: one for the training data and another for the test data. Each plot shows the clusters based on `Longitude` and `Latitude`.

👉 Therefore, KNN classifier seems like a good choice for coordination calculations.

👨‍💼 Preparing the dataset for training

We prepare the dataset for KNN model training, create and train the KNN model, and save the KNN model for future use.

📈 Confusion Matrix and Classification Report

We evaluate the KNN model's performance using a confusion matrix and a classification report.

🙋‍♂️ Conclusions

✅ Precision: The model achieved high precision for all classes, indicating that it correctly identified most positive samples for each class.

✅ Recall (Sensitivity): The model achieved high recall for all classes, indicating that it correctly identified a large proportion of positive samples for each class.

✅ F1-Score: The F1-Score, which balances precision and recall, is high for all classes, suggesting a well-rounded model performance.

✅ Support: This represents the number of true samples for each class in the dataset.

✅ Accuracy: The model achieved 100% accuracy overall, meaning that all samples were correctly classified in the test dataset.

In summary, the model demonstrates exceptional performance with high precision, recall, and F1-Score for all classes. It achieves perfect accuracy on the test dataset, indicating its effectiveness in classifying the data.

🧐 Cross Check for KNN

We perform cross-validation to further validate the KNN model's performance.

🧙‍♂️ Conclusions

✅ Data Overview:

- The dataset consists of 20,640 entries and 9 columns.
- Columns include features such as 'MedInc,' 'HouseAge,' 'AveRooms,' 'AveBedrms,' 'Population,' 'AveOccup,' 'Latitude,' 'Longitude,' and the target variable 'MedHouseVal.'
- All columns have non-null values, and they are of float64 data type.

✅ K-Means Clustering:

- The K-Means clustering algorithm was applied to the dataset with 6 clusters.
- The results include latitude, longitude, 'MedInc,' and the cluster label for some sample data points.

✅ K-Nearest Neighbors (KNN) Classification:

- The KNN classifier was used with k=3 neighbors.
- The model achieves high accuracy, precision, recall, and F1-score for all classes, indicating excellent performance.

✅ Cross-Validation:

- Cross-validation was performed with 5 folds, resulting in accuracy scores for each fold.
- The model's performance is stable across folds.

In summary, the K-Means clustering was used to group data points into clusters, and the KNN classifier achieved excellent performance in classifying the data points into classes. The model demonstrated high accuracy and consistency across cross-validation folds.
