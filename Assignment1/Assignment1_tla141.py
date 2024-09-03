import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


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


# splitte lables, splitte hver label 80:20, tilslutt merge 80 med 20 og 20 med 80

# We were told that using SKlearn for train_test_split was allowed.

# Separating the labels and storing in new variables
X = fdf[fdf["label"]==1]
y = fdf[fdf["label"]==0]
 
# Splitting the dataset 80/20
X_train, X_test = train_test_split(X,test_size=0.2, random_state=0)
y_train, y_test = train_test_split(y, test_size=0.2, random_state=0)

print(f"\nxtrain: 80% of pop\n {X_train} \n\nxtest: 20% of pop \n {X_test} \n")
print(f"ytrain: 80% of classical \n {y_train} \n\nytest:20% of classical \n {y_test} \n")

# Joining the two test sets and the two training sets
test_sets = pd.concat((X_test, y_test), axis=0)
train_sets = pd.concat((X_train, y_train))

print(f"\n test sets 20%: \n {test_sets} \n\ntrain sets 80%: \n {train_sets} \n")

"""1d"""
# print(f"\n\nProblem 1d\n")