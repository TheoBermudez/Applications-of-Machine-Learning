'''
Your task is to build a classification model that predicts the edibility of mushrooms (class variable in the dataset). 
You have been provided with a dataset as a "mushrooms(1).csv" file. Here is a description of the attributes:
Here is a description of the attributes:

Attribute description:

1. cap-shape: bell=b, conical=c, convex=x, flat=f, knobbed=k, sunken=s
2. cap-surface: fibrous=f, grooves=g, scaly=y, smooth=s
3. cap-color: brown=n, buff=b, cinnamon=c, gray=g, green=r, pink=p, purple=u, red=e, white=w, yellow=y
4. bruises: bruises=t, no=f
5. odor: almond=a, anise=l, creosote=c, fishy=y, foul=f, musty=m, none=n, pungent=p, spicy=s
6. gill-attachment: attached=a, descending=d, free=f, notched=n
7. gill-spacing: close=c, crowded=w, distant=d
8. gill-size: broad=b, narrow=n
9. gill-color: black=k, brown=n, buff=b, chocolate=h, gray=g, green=r, orange=o, pink=p, purple=u, red=e, white=w, yellow=y
10. stalk-shape: enlarging=e, tapering=t
11. stalk-root: bulbous=b, club=c, cup=u, equal=e, rhizomorphs=z, rooted=r, missing=?
12. stalk-surface-above-ring: fibrous=f, scaly=y, silky=k, smooth=s
13. stalk-surface-below-ring: fibrous=f, scaly=y, silky=k, smooth=s
14. stalk-color-above-ring: brown=n, buff=b, cinnamon=c, gray=g, orange=o, pink=p, red=e, white=w, yellow=y 
15. stalk-color-below-ring: brown=n, buff=b, cinnamon=c, gray=g, orange=o, pink=p, red=e, white=w, yellow=y 
16. veil-type: partial=p, universal=u
17. veil-color: brown=n, orange=o, white=w, yellow=y
18. ring-number: none=n, one=o, two=t
19. ring-type: cobwebby=c, evanescent=e, flaring=f, large=l, none=n, pendant=p, sheathing=s, zone=z
20. spore-print-color: black=k, brown=n, buff=b, chocolate=h, green=r, orange=o, purple=u, white=w, yellow=y 
21. population: abundant=a, clustered=c, numerous=n, scattered=s, several=v, solitary=y
22. habitat: grasses=g, leaves=l, meadows=m, paths=p, urban=u, waste=w, woods=d
23. class: p = poisonous, e=edible

Build a classification tree using random_state = 2023, training partition = 0.75, stratify = y, max_depth = 6
'''

# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
import matplotlib.pyplot as plt

# Load the dataset
mushrooms = pd.read_csv('mushrooms(1).csv')

# Convert all categorical variables into dummy/indicator variables before splitting
mushrooms_encoded = pd.get_dummies(mushrooms, drop_first=True)

# Separate the features and the target variable
X = mushrooms_encoded.drop('class_p', axis=1)
y = mushrooms_encoded['class_p']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=2023, stratify=y)

# Initialize the Decision Tree Classifier with max_depth=6
clf = DecisionTreeClassifier(max_depth=6, random_state=2023)
clf.fit(X_train, y_train)

# A. Print the confusion matrix. Also visualize the confusion matrix using ConfusionMatrixDisplay from sklearn.metrics
y_pred = clf.predict(X_test)
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=['Edible', 'Poisonous'])
disp.plot(cmap='Blues', values_format='d')
plt.title('Confusion Matrix')
plt.show()

# B. What was the accuracy on the training partition?
train_accuracy = accuracy_score(y_train, clf.predict(X_train))
print(f"Training Accuracy: {train_accuracy:.4f}")

# C. What was the accuracy on the test partition?
test_accuracy = accuracy_score(y_test, y_pred)
print(f"Testing Accuracy: {test_accuracy:.4f}")

# D. Show the classification tree
plt.figure(figsize=(20, 10))
plot_tree(clf, filled=True, feature_names=X.columns, class_names=['Edible', 'Poisonous'])
plt.title('Decision Tree Classifier for Mushroom Dataset')
plt.show()

# E. List the top three most important features in your decision tree for determining toxicity
feature_importances = pd.Series(clf.feature_importances_, index=X.columns).sort_values(ascending=False)
print("Top three most important features:")
print(feature_importances.head(3))

# F. Classify the mushroom sample
new_sample = pd.DataFrame([{'cap-shape_x': 1, 'cap-surface_s': 1, 'cap-color_n': 1, 'bruises_t': 1, 'odor_y': 1,
                            'gill-attachment_f': 1, 'gill-spacing_c': 1, 'gill-size_n': 1, 'gill-color_k': 1,
                            'stalk-shape_e': 1, 'stalk-root_e': 1, 'stalk-surface-above-ring_s': 1,
                            'stalk-surface-below-ring_s': 1, 'stalk-color-above-ring_w': 1, 'stalk-color-below-ring_w': 1,
                            'veil-type_p': 1, 'veil-color_w': 1, 'ring-number_o': 1, 'ring-type_p': 1,
                            'spore-print-color_r': 1, 'population_s': 1, 'habitat_u': 1}])

new_sample = new_sample.reindex(columns=X_train.columns, fill_value=0)

new_sample_prediction = clf.predict(new_sample)
print(f"Class Prediction for New Mushroom: {'Poisonous' if new_sample_prediction[0] == 1 else 'Edible'}")
