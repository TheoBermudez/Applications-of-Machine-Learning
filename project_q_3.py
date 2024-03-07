'''
Load the “UniversalBank(1).csv" (this dataset is taken from the website of the book "Data mining for business intelligence" by Shmueli, Patel and Bruce, 1st ed, Wiley 2006).
The data set provides information about many people and our goal is to build a model to classify the cases into those who will accept the offer of a personal loan and those who will reject it. 
In the data, a zero in the Personal loan column indicates that the concerned person rejected the offer and a one indicates that the person accepted the offer. 
Answer the following questions:
'''

# import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

# A. What is the target variable?
#    Load the data from the file "UniversalBank(1).csv"
df = pd.read_csv('UniversalBank(1).csv')
X = df.drop('Personal Loan', axis=1)
y = df['Personal Loan']

# B. Ignore the variables Row and Zip code
df = df.drop(['Row', 'ZIP Code'], axis=1)

# C. Partition the data 75/25, random_state = 2023, stratify = y
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=2023, stratify=y)

# D. How many of the cases in the training partition represented people who accepted the offer of a personal loan?
accepted_cases_train = y_train.sum()
print(f"Number of accepted cases in the training partition: {accepted_cases_train}")

# E. Plot the classification tree. Use entropy criterion. max_depth = 5, random_state = 2023
dt_classifier = DecisionTreeClassifier(random_state=2023, criterion='entropy', max_depth=5)
dt_classifier.fit(X_train, y_train)
plt.figure(figsize=(15, 10))
plot_tree(dt_classifier, feature_names=X.columns, class_names=['0', '1'], filled=True, rounded=True)
plt.show()

'''
F. On the testing partition, how many acceptors did the model classify as non-acceptors?
G. On the testing partition, how many non-acceptors did the model classify as acceptors?
'''
y_pred = dt_classifier.predict(X_test)
false_non_acceptors = ((y_test == 1) & (y_pred == 0)).sum()
false_acceptors = ((y_test == 0) & (y_pred == 1)).sum()
print(f"False non-acceptors: {false_non_acceptors}")
print(f"False acceptors: {false_acceptors}")

# H. What was the accuracy on the training partition?
train_accuracy = dt_classifier.score(X_train, y_train)
print(f"Training Accuracy: {train_accuracy}")

# I. What was the accuracy on the test partition?
test_accuracy = dt_classifier.score(X_test, y_test)
print(f"Test Accuracy: {test_accuracy}")
