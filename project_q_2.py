# Load the file "Stores.csv". Perform k-means clustering:

# Import necessary libraries.
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load the file "Stores.csv".
df = pd.read_csv('Stores.csv')

# A. Perform the necessary data preparation for the stores dataframe:
#    Extract 'Store' column and save it in a separate variable, then drop it from the dataframe.
#    Standardize the dataset (use StandardScaler).
store_names = df['Store']
df = df.drop('Store', axis=1)
scaler = StandardScaler()
df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

# B. Run k-means for k ranging from 1 to 10. random_state = 2023, n_init='auto'.
inertias = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=2023, n_init='auto')
    kmeans.fit(df)
    inertias.append(kmeans.inertia_)

# C. Plot the inertias vs k.
plt.figure(figsize=(12, 6))
plt.plot(range(1, 11), inertias, marker='o')
plt.title('Inertias vs. K')
plt.xlabel('K')
plt.ylabel('Inertia')
plt.show()

# D. What is the best k?
best_k = inertias.index(min(inertias)) + 1

# E. What cluster does this store belong to?
new_data = [6.3, 3.5, 2.4, 0.5]
new_data = scaler.transform([new_data])
cluster = kmeans.predict(new_data)[0]
print(f'The store belongs to cluster {cluster}')

# F. Now add the 'Store' and 'Cluster' columns to the original dataframe. Display the dataframe.
df['Store'] = store_names
df['Cluster'] = kmeans.labels_
print(df)

# G. Plot a histogram of cluster number.
df['Cluster'].hist()
plt.title('Histogram of Cluster Numbers')
plt.xlabel('Cluster Number')
plt.ylabel('Frequency')
plt.show()
