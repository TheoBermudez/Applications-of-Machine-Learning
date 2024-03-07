# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# 1. Create a DataFrame “diabetes_knn” to store the diabetes data and set the option to display all columns without any restrictions.
pd.set_option('display.max_columns', None)
diabetes_knn = pd.read_csv('diabetes.csv')  

# 2. Determine the dimensions of the “diabetes_knn” dataframe.
print('DataFrame dimensions:', diabetes_knn.shape)

# 4. Replace all 0 values with the mean of the column (except for "Pregnancies" column)
columns_to_replace = diabetes_knn.columns.drop('Pregnancies').drop('Outcome')  
for column in columns_to_replace:
    diabetes_knn[column] = diabetes_knn[column].replace(0, diabetes_knn[column].mean())

# Create the Feature Matrix and Target Vector
X = diabetes_knn.drop('Outcome', axis=1)
y = diabetes_knn['Outcome'].astype('int')  

# 5. Standardize the attributes of the Feature Matrix
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)  

# 6. Split the Feature Matrix and Target Vector into Train A (70%) and Train B (30%) sets. Use random_state=2023, and stratify based on the Target vector.
X_trainA, X_trainB, y_trainA, y_trainB = train_test_split(X_scaled_df, y, test_size=0.3, random_state=2023, stratify=y)

# 7. Develop a KNN model and obtain KNN score (accuracy) for Train A and Train B data for k’s values ranging between 1 to 8.
trainA_scores = []
trainB_scores = []
for k in range(1, 9):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_trainA, y_trainA)
    trainA_scores.append(knn.score(X_trainA, y_trainA))
    trainB_scores.append(knn.score(X_trainB, y_trainB))

# 8. Plot a graph of Train A and Train B score and determine the best value of k.
plt.figure(figsize=(12, 6))
plt.plot(range(1, 9), trainA_scores, marker='o', label='Train A Score')
plt.plot(range(1, 9), trainB_scores, marker='o', label='Train B Score')
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy')
plt.title('KNN Score for Different Values of k')
plt.legend()
plt.show()

best_k = trainB_scores.index(max(trainB_scores)) + 1
print(f'Best value of k: {best_k}')

# 9. Use the best k to train the model and evaluate it on Train B
knn_best = KNeighborsClassifier(n_neighbors=best_k)
knn_best.fit(X_trainA, y_trainA)
y_predB = knn_best.predict(X_trainB)
cm = confusion_matrix(y_trainB, y_predB)
print('Confusion Matrix for Train B:')
ConfusionMatrixDisplay(confusion_matrix=cm).plot()
plt.title('Confusion Matrix for Train B')
plt.show()

# 10. Predict the Outcome for a new data point
new_data = [[6, 140, 60, 12, 300, 24, 0.4, 45]]
new_data_df = pd.DataFrame(new_data, columns=X.columns)
new_data_scaled = scaler.transform(new_data_df)  
prediction = knn_best.predict(new_data_scaled)
print(f'Prediction for the new data: {"Diabetic" if prediction[0] == 1 else "Not Diabetic"}')
