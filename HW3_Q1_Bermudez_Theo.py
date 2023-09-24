# Theo Bermudez
# ITP 449
# HW3
# Question 1

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# 1. Create a dictionary in Python, which contains this information. Then define a DataFrame in pandas using this dictionary.
# Hint: Use np.nan to define NaN entries
data = {'attempts': [1, 3, 2, 3, 2, 3, 1, 1, 2, 1],
        'name': ['Anastasia', 'Dima', 'Katherine', 'James', 'Emily', 'Michael', 'Matthew', 'Laura', 'Kevin', 'Jonas'],
        'qualify': ['yes', 'no', 'yes', 'no', 'no', 'yes', 'yes', 'no', 'no', 'yes'],
        'score': [12.5, 9.0, 16.5, np.nan, 9.0, 20.0, 14.5, np.nan, 8.0, 19.0]}
df = pd.DataFrame(data)
print(df)

# 2. Write a single-line code in Python to print the name and the attempts of qualified contestants.
print(df[['name', 'attempts']][df['qualify'] == 'yes'])

# 3. Write a single-line code in Python to print the name and the score of those contestants, who qualified with a single attempt.
# Hint: Check out np.logical_and() in numpy
print(df[['name', 'score']][np.logical_and(df['qualify'] == 'yes', df['attempts'] == 1)])

# 4. Write a single-line code in Python to replace all the NaN values with Zero's in the score column of a dataframe. Then print the dataframe to confirm the change.
# Hint: Use np.isnan() in numpy
df['score'] = df['score'].apply(lambda x: 0 if np.isnan(x) else x)
print(df)

# 5. Write a single-line code in Python to print the dataframe such that it is sorted the by attempts in ascending order (and score in descending order if 2 contestants have the same number of attempts.)
# Note: Make sure to leave inplace=False
print(df.sort_values(by=['attempts', 'score'], ascending=[True, False]))
