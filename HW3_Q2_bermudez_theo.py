# Theo Bermudez
# ITP 449
# HW3
# Question 2

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# 1. Read the csv file using Pandas. Store the output into a dataframe named DF. Then print DF
df = pd.read_csv('Trojans_roster.csv')
print(df)

# 2. You notice that the index is 0 ... 109. There is a column #. Set the index of the dataframe to #. In other words, make the player’s number column the index of DF. Print DF to make sure that the change happened.
# Hint: Check out set_index() method for dataframe. Make sure to make the change inplace=True
df.set_index('#', inplace=True)
print(df)

# 3. Remove the ‘LAST SCHOOL’ and ‘MAJOR’ columns from the dataframe. Print DF to make sure that the change happened.
df.drop(['LAST SCHOOL', 'MAJOR'], axis=1, inplace=True)
print(df)

# 4. Write a single-line code in Python to print the names of all the Quarterbacks in the team.
print(df['NAME'][df['POS.'] == 'QB'])

# 5. Write a single-line code in Python to print the name, position, height, and weight of the tallest player in the team.
print(df[['NAME', 'POS.', 'HT.', 'WT.']][df['HT.'] == df['HT.'].max()])

# 6. Write a single-line code in Python to print how many players are local (i.e., their hometown is ‘LOS ANGELES, CA’). Note that the answer is a number.
print(len(df[df['HOMETOWN'] == 'LOS ANGELES, CA']))

# 7. Write a single-line code in Python to print the info of 3 heaviest players.
print(df.sort_values(by='WT.', ascending=False).head(3))

# 8. Define a new column for DF named BMI, which contains the BMI of the players. Print DF to make sure that the change happened.
# BMI = 703 × Weight (𝑙𝑏) Height2 (𝑖𝑛)
df['BMI'] = 703 * df['WT.'] / (df['HT.'] * df['HT.'])
print(df)

# 9. Write single-line codes in Python to print the mean and median of players’ height, weight, and BMI.
print(df[['HT.', 'WT.', 'BMI']].mean())
print(df[['HT.', 'WT.', 'BMI']].median())

# 10. Write single-line codes in Python to print the mean and median of players’ height, weight, and BMI for each position.
# Hint: Check out groupby() method for dataframe.
print(df.groupby('POS.')[['HT.', 'WT.', 'BMI']].mean())
print(df.groupby('POS.')[['HT.', 'WT.', 'BMI']].median())

# 11. Write a single-line code in Python to print the number of players in each position.
# Hint: Check out count() method for dataframe.
print(df.groupby('POS.').count())

# 12. Write a single-line code in Python to print the names of the players whose BMI is below the team average (mean).
print(df['NAME'][df['BMI'] < df['BMI'].mean()])

# 13. Write a single-line code in Python to print all the unique players’ numbers. (unique numbers in DF index)
# Hint: Check out unique() method for dataframe.
print(df.index.unique())