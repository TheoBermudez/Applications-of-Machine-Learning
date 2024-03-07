'''
Load the data from the file "diabetes.csv" and create a KNN model for diabetes prediction. Explore the factors in the dataset.
'''

# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# 1. Create a DataFrame “diabetes_knn” to store the diabetes data and set option to display all columns without any restrictions on the number of columns displayed.
pd.set_option('display.max_columns', None)
diabetes_knn = pd.read_csv('diabetes.csv')  

# 2. Determine the dimensions of the “diabetes_knn” dataframe.
print('DataFrame dimensions:', diabetes_knn.shape)

# 3. Update the DataFrame to account for missing values if needed.
columns_to_replace = diabetes_knn.columns.drop('Pregnancies').drop('Outcome')  
for column in columns_to_replace:
    diabetes_knn[column] = diabetes_knn[column].replace(0, diabetes_knn[column].mean())

# 4. Create the Feature Matrix and Target Vector.
X = diabetes_knn.drop('Outcome', axis=1)
y = diabetes_knn['Outcome'].astype('int')  

# 5. Standardize the attributes of Feature Matrix (https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)  

# 6. Split the Feature Matrix and Target Vector into train A (70%) and train B sets (30%). Use random_state=2023, and stratify based on Target vector.
X_trainA, X_trainB, y_trainA, y_trainB = train_test_split(X_scaled_df, y, test_size=0.3, random_state=2023, stratify=y)

# 7. Develop a KNN based model and obtain KNN score (accuracy) for train A and train B data for k’s values ranging between 1 to 8.
trainA_scores = []
trainB_scores = []
for k in range(1, 9):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_trainA, y_trainA)
    trainA_scores.append(knn.score(X_trainA, y_trainA))
    trainB_scores.append(knn.score(X_trainB, y_trainB))

# 8. Plot a graph of train A and train B score and determine the best value of k.
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

''' 
9. Display the score of the model with best value of k. Also print and plot the confusion matrix for Train B, using Train A set as the reference
   set for training.
'''
knn_best = KNeighborsClassifier(n_neighbors=best_k)
knn_best.fit(X_trainA, y_trainA)
y_predB = knn_best.predict(X_trainB)
cm = confusion_matrix(y_trainB, y_predB)
print('Confusion Matrix for Train B:')
ConfusionMatrixDisplay(confusion_matrix=cm).plot()
plt.title('Confusion Matrix for Train B')
plt.show()

'''
10. Predict the Outcome for a person with pregnancies=6, glucose=140, blood pressure=60, skin thickness=12, insulin=300, BMI=24,
    diabetes pedigree=0.4, age=45.
'''
new_data = [[6, 140, 60, 12, 300, 24, 0.4, 45]]
new_data_df = pd.DataFrame(new_data, columns=X.columns)
new_data_scaled = scaler.transform(new_data_df)  
prediction = knn_best.predict(new_data_scaled)
print(f'Prediction for the new data: {"Diabetic" if prediction[0] == 1 else "Not Diabetic"}')
