# For this problem, you will be doing classification with KNN. The goal is to predict the quality of wine given the other attributes.
 
# Import necessary libraries.
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# A. Load the data from the file "winequality(1).csv".
df = pd.read_csv('winequality(1).csv')

# B. Standardize all variables other than Quality. (use StandardScaler)
scaler = StandardScaler()
df[df.columns.difference(['Quality'])] = scaler.fit_transform(df[df.columns.difference(['Quality'])])

# C. Partition the dataset (Use random_state = 2023, Partitions 60/20/20, stratify = y).
X = df.drop('Quality', axis=1)
y = df['Quality']
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=2023, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.25, random_state=2023, stratify=y_temp)

'''
D. Build a KNN classification model to predict Quality based on all the remaining numeric variables.
E. Iterate on K ranging from 1 to 30. Plot the accuracy for the train A and train B datasets.
'''
train_scores = []
val_scores = []
for k in range(1, 31):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    train_scores.append(accuracy_score(y_train, knn.predict(X_train)))
    val_scores.append(accuracy_score(y_val, knn.predict(X_val)))

plt.figure(figsize=(12, 6))
plt.plot(range(1, 31), train_scores, marker='o', label='Train')
plt.plot(range(1, 31), val_scores, marker='o', label='Validation')
plt.title('Accuracy vs. K Value')
plt.xlabel('K')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# F. Which value of k produced the best accuracy in the train A and train B data sets?
best_k = val_scores.index(max(val_scores)) + 1

'''
G. Generate predictions for the test partition with the chosen value of k. 
   Print and plot the confusion matrix of the actual vs predicted wine quality.
'''
knn = KNeighborsClassifier(n_neighbors=best_k)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)
ConfusionMatrixDisplay(cm).plot()
plt.show()

# H. Print the test dataframe with the added columns “Quality” and “Predicted Quality”.
df_test = X_test.copy()
df_test['Quality'] = y_test
df_test['Predicted Quality'] = y_pred
print(df_test)

# I. Print the accuracy of model on the test dataset.
print("Test Accuracy: ", accuracy_score(y_test, y_pred))
