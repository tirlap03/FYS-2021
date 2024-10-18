import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
import seaborn as sns

# Read the CSV file into a DataFrame (assuming there's no header row)
dfd = pd.read_csv("data_problem2.csv", header=None)

# Print DataFrame information
# print(dfd.info())
# print("Shape of DataFrame:", dfd.shape)
# print(dfd)

# Extract the first row (since dfd.shape is (1, 3600), we're interested in the first row)
# features = dfd.to_numpy()  # Convert the first row to a NumPy array

features = dfd.iloc[0, :].to_numpy()  # First row (features)
labels = dfd.iloc[1, :].to_numpy()  # Second row (labels)



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

""" Estimating parameters using maximum liklihood estimation MLE for each class """
def parameter_estimation(X_train, y_train):
    classes = np.unique(y_train)
    params = {}

    for c in classes:
        X_c = X_train[y_train == c]

        mean_c = np.mean(X_c)
        var_c = np.var(X_c)

        params[c] = {
            "mean": mean_c,
            "var": var_c,
            "prior": len(X_c) / len(X_train)
        }

        # print(f"\n class parameters {c}\n mean {mean_c:.4f}\n var {var_c:.4f}\n prior prob {params[c]}")

    return params

def gausian_likelihood(x, mean, var):
    return (1 / np.sqrt(2*np.pi*var)) * np.exp(-(x-mean)**2 / (2*var))

def bayes_classifier(X, params):
    predictions = []

    posterios = {}
    for x in X:
        for c, param in params.items():
            likelihood = gausian_likelihood(x, param["mean"], param["var"])
            posterior = likelihood * param["prior"]
            posterios[c] = posterior
        
        pred_class = max(posterios.items(), key=lambda x: x[1])[0]
        predictions.append(pred_class)

    return np.array(predictions)


params = parameter_estimation(X_train, y_train)
predictions = bayes_classifier(X_test, params)

accuracy = np.mean(predictions == y_test)
print(f"\nAccuracy: {accuracy}")


# Create a DataFrame for the predictions
results_df = pd.DataFrame({
    'Features': X_test,
    'True Labels': y_test,
    'Predictions': predictions
})

# Identify misclassifications
results_df['Correct'] = results_df['True Labels'] == results_df['Predictions']

# Now we can plot the histogram using the correct and misclassified data
plt.figure(figsize=(10, 6))

# Plotting correctly classified data in one color
sns.histplot(data=results_df[results_df['Correct']], 
            x='Features', 
            bins=30, 
            color='lightblue', 
            edgecolor='black', 
            label='Correctly Classified', 
            alpha=0.5)

# Plotting misclassified data in another color
sns.histplot(data=results_df[~results_df['Correct']], 
            x='Features', 
            bins=30, 
            color='red', 
            edgecolor='black', 
            label='Misclassified', 
            alpha=0.5)

plt.title('Histogram of Features with Classification Results')
plt.xlabel('Feature Value')
plt.ylabel('Frequency')
plt.legend()
plt.tight_layout()

# Show the plot
plt.savefig("Histogram_Classification_Results.png")
plt.show()