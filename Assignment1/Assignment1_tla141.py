import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import math


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

"""1d: Bonus"""
# print(f"\n\nProblem 1d\n")

""""""""""""""""""""""""""""""""""""""""""""""""""""""

"""2a"""
print(f"\n\nProblem 2a\n")

class logistic_classifer ():
    def __init__(self, lr, epochs, bias):
        self.lr = lr                    #learning rate (lr = 0.1)
        self.epochs = epochs            # epochs = 10    
        self.weights = np.zeros(3)      # w0, w1, w2
        self.bias = bias                # bias = 1
        
    def linear_regression(self, feature1, feature2):
        lin_reg = np.sum(self.weights[0]*1 + self.weights[1]*feature1 + self.weights[2]*feature2)
        # print(f"\nlinear regression: {lin_reg}\n")
        return lin_reg

    def zigmoid(self, lin_reg):
        lin_reg = np.clip(lin_reg, -500, 500)
        z = 1/(1+np.exp(-lin_reg))          # kan bruke clip() for å fikse overflow
        # print(f"\nzigmoid : {z}\n")
        return z

    def loss_func(self, pred, val):
        epsilon = 1e-10
        pred = np.clip(pred, epsilon, 1-epsilon)
        # kan clip() pred slik at den aldri blir 1 eller 0, ved å bruke en liten epsilon = 1e-10 
        loss = -(val*np.log(pred)) + ((1-val)*np.log(1-pred))       
        # print(f"\nloss func: {loss}\n")
        return loss     # loss.mean(), men hvorfor?
    
    def SGD(self, X_train, y_train):
        for epoch in range(self.epochs):
            for i in range(len(X_train)):                   # Iterating over all datapoints of the training dataset.
                # Extracting the values of the current datapoint 
                feature1 = X_train.iloc[i]["liveness"]
                feature2 = X_train.iloc[i]["loudness"]

                # usikker på om det er y_train som skal brukes her (eller om det er test-verdier til sammenligning)
                target = y_train.iloc[i]

                lin_reg = self.linear_regression(feature1, feature2)
                prediction = self.zigmoid(lin_reg)

                # Calculating error, and updating the weights
                error = prediction - target
                self.weights[0] -= self.lr *error               # this is the bias
                self.weights[1] -= self.lr *error *feature1
                self.weights[2] -= self.lr *error *feature2
            
            if epoch % 100 == 0:
                print(f'Epoch {epoch}, Loss {self.loss_func(self.zigmoid(self.linear_regression(X_train["liveness"], X_train["loudness"])), y_train)}')


def plotting_separation_line(self, X, y):
    x_min, x_max = X["liveness"].min() - 1, X["liveness"].max() + 1
    y_min, y_max = X["loudness"].min() - 1, X["loudness"].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))

    Z = self.sigmoid(self.linear_regression(xx.ravel(), yy.ravel()))
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.RdYlBu)
    plt.scatter(X['liveness'], X['loudness'], c=y, edgecolors='k', cmap=plt.cm.RdYlBu)
    plt.xlabel('Liveness')
    plt.ylabel('Loudness')
    plt.title('Decision Boundary')
    plt.colorbar()
    plt.show()

def test_predict(test_predictions, threshold = 0.5):
    for pred in test_predictions:   
        if pred > threshold:
            return 1
        else:
            return 0


if __name__=="__main__":

    log_class = logistic_classifer(lr = 0.1, epochs=1, bias=1)

    log_class.SGD(train_sets, train_sets["label"])

    test_lin_reg = log_class.linear_regression(test_sets["liveness"], test_sets["loudness"])
    test_predictions = log_class.zigmoid(test_lin_reg)
    test_predictions = [test_predictions]

    # kan hende test_prediction må være en list eller array slik at den itereres.
    test_preds_binary = test_predict(test_predictions)

    accuracy = np.mean(test_preds_binary == test_sets["label"])

    print(f"\n\nModel Accuracy: {accuracy}")

    # assert len(test_preds_binary) == len(X_train)

    plt.ion()

    x = np.linspace(fdf['liveness'].min(), fdf['liveness'].max(), 100)
    y = np.linspace(fdf['loudness'].min(), fdf['loudness'].max(), 100)

    # Create a meshgrid for x and y
    x_grid, y_grid = np.meshgrid(x, y)

    # Compute z as a linear combination of x and y
    z = x_grid + y_grid

    # Apply the sigmoid function to z
    sigmoid_values = log_class.zigmoid(z)

    # Create a 3D plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the surface for the sigmoid function
    ax.plot_surface(x_grid, y_grid, sigmoid_values, cmap='viridis')

    # Add labels and title
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('Sigmoid(x + y)')
    ax.set_title('3D Sigmoid Function (S-curve)')
    plt.savefig('scatter_plot.png')

    plt.show(block=True)


"""2b"""
print(f"\n\nProblem 2b\n")

"""2c: Bonus"""
# print(f"\n\nProblem 2c\n")