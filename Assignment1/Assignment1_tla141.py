import pandas as pd
import numpy as np


"""1a"""
print(f"\n\nProblem 1a\n")

df = pd.read_csv("SpotifyFeatures.csv")

print(df.info())



"""1b"""
print(f"\n\nProblem 1b\n")

# Looking at selected columns
fdf = df[df['genre'].isin(['Pop', 'Classical'])]

# Creating a label for the two rows
label_dict = {
    "Pop": 1,
    "Classical": 0
}
fdf = fdf.copy()
# Mapping the lables to the coresponding rows
fdf["label"] = fdf["genre"].map(label_dict)

# Counting how many elements each label contains
count_each = fdf.value_counts("label")
print(f"\nCounting each value (1.0 is pop, 0.0 is classical): {count_each}\n")

fdf = fdf[["liveness", "loudness", "label"]]
print(fdf)



"""1c"""
print(f"\n\nProblem 1c\n")
# First array contains songs as rows and songs features as columns in a matrix
arr1 = fdf[["liveness", "loudness"]].to_numpy()
print(f"The matrix where songs are rows, and songs features are columns: \n {arr1} \n")

arr2 = fdf["label"].to_numpy()
print(f"The vector containing the labels: \n {arr2} \n")

# Splitting the dataset
split = int(0.8*len(fdf))
training = fdf.iloc[:split]
test = fdf.iloc[split:]

print(f"training: \n {training}\n")
print(f"test: \n {test}\n")


"""1d"""
print(f"\n\nProblem 1d\n")