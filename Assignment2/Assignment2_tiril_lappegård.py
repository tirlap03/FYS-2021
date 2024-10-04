import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
import seaborn as sns

# Read the CSV file into a DataFrame (assuming there's no header row)
dfd = pd.read_csv("data_problem2.csv", header=None)

# Print DataFrame information
print(dfd.info())
print("Shape of DataFrame:", dfd.shape)
print(dfd)

# Extract the first row (since dfd.shape is (1, 3600), we're interested in the first row)
# features = dfd.to_numpy()  # Convert the first row to a NumPy array

features = dfd.iloc[0, :].to_numpy()  # First row (features)
labels = dfd.iloc[1, :].to_numpy()  # Second row (labels)

print(features)
print(features.shape)


# Combine the features and labels into a DataFrame for Seaborn
df_combined = pd.DataFrame({
    'Features': features,
    'Labels': labels
})

# Plot the histogram using Seaborn
plt.figure(figsize=(10, 6))
sns.histplot(data=df_combined, x='Features', hue='Labels', bins=30, edgecolor='black', palette='Set1')

plt.title('Histogram of Features with Label Separation')
plt.xlabel('Feature Value')
plt.ylabel('Frequency')
plt.tight_layout()

# Show the plot
plt.savefig("Histogram1.png")
# plt.show()



###########################################################################################################
"Testing into train and test sets"

# print(df_combined)

df_array = df_combined.to_numpy()
np.random.shuffle(df_array)

# print(df_array)

features_random = df_array[:, 0]
labels_random = df_array[:, 1]

# print(features_random)
# print(labels_random)

X_train, X_test = train_test_split(features_random, test_size=0.2, random_state=0)
y_train, y_test = train_test_split(labels_random, test_size=0.2, random_state=0)

# print(f"X_train: 80% of the train set \n{X_train}\nWith the shape {X_train.shape}\n\n X_test: 20% of the train set \n{X_test}\nWith the shape of {X_test.shape}\n\n")
# print(f"y_train: 80% of the test set \n{y_train}\nWith the shape of {y_train.shape}\n\n y_test:20% of the test set \n{y_test}\nWith the shape of {y_test.shape}\n\n")