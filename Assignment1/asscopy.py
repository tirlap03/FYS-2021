import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import numdifftools as nd

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

pop = fdf[fdf["label"]==1]                          # Extracting the rows categorized as pop genre
classical = fdf[fdf["label"]==0]                    # Extracting the rows categorized as classical genre
print(f"pop matrix \n{pop}\nshape {pop.shape}\nclassical matrix \n{classical}\n shape {classical.shape}\n")

big_a = pd.concat((pop, classical)).to_numpy()      # Joining the two matrixes together into one big matrix
print(f"big a \n{big_a}\n shape \n{big_a.shape}\n")

np.random.shuffle(big_a)                            # Shuffling the values inside the matrix randomly, to not feed the machine only pop then only classical
print(f"shuffled big a\n{big_a}\n")

features = big_a[:, :2]                             # Choosing the two first columns from the shuffled matrix to be the features.
labell = big_a[:, 2]                                # Choosing the last column from the shuffled matrix to be the labels
print(f"The matrix where songs are rows, and songs features are columns: \n {features} \n the shape {features.shape}\n")
print(f"The vector containing the labels: \n {labell} \n the shape {labell.shape}\n")

X_train, X_test = train_test_split(features, test_size=0.2, random_state=0)         # Splitting the features into 80% train and 20% test
y_train, y_test = train_test_split(labell, test_size=0.2, random_state=0)           # Splitting the labels into 80% train and 20% test
print(f"X_train: 80% of the train set \n{X_train}\n X_test: 20% of the train set \n{X_test}\n")
print(f"y_train: 80% of the test set \n{y_train}\n y_test:20% of the test set \n{y_test}\n")
print(f"shapes: \nx_tr {X_train.shape}\nx_te {X_test.shape}\ny_tr {y_train.shape}\ny_te {y_test.shape}\n")


"""1d: Bonus"""
# print(f"\n\nProblem 1d\n")

""""""""""""""""""""""""""""""""""""""""""""""""""""""

"""2a"""
print(f"\n\nProblem 2a\n")
class logistic_classifer ():
    def __init__(self, lr=0.01, epochs=100, bias=None):
        self.lr = lr                    #learning rate
        self.epochs = epochs            # epochs = 10    
        self.weights = None             
        self.bias = None                
        

    def zigmoid(self, lin_reg):
        lin_reg = np.clip(lin_reg, -500, 500)       # Clipping the values below and over 500, to avoid overflow
        z = 1/(1+np.exp(-lin_reg))                  # Calculating the sigmoid function    
        return z    
    
    
    def SGD(self, X_train, y_train):
        self.bias = 0                               # Initializing the bias as 0
        samples, features = X_train.shape           # Initializing the number of samples and features as the rows and columns of the matrix
        self.weights = np.zeros(features)           # Initializing the weights as 0 (as many as there are features, here there is two)
        for epoch in range(self.epochs):
            for i in range(len(X_train)):             
                lin_pred = np.dot(X_train, self.weights) + self.bias        # Calculating the predictions of the linear regression
                predictions = self.zigmoid(lin_pred)                        # Calculating the actual predictions, which is the sigmoid function

                # Calculating the gradient
                dw = (1/samples) * np.dot(X_train.T, (predictions - y_train))       
                db = (1/samples) * np.sum(predictions-y_train)

                # Calculating gradient, and updating the weights
                self.weights = self.weights - self.lr *dw
                self.bias = self.bias - self.lr*db

            if epoch % 100 == 0:
                print(f'Epoch {epoch}')
                           
    # Predicting the values of y_train based on the training of X_train
    def predict_test(self, X_train):
        linear_pred = np.dot(X_train, self.weights) + self.bias
        # print(f"linreg: {lin_reg}\n\n")
        y_pred = self.zigmoid(linear_pred)

        class_pred = [0 if y<=0.5 else 1 for y in y_pred]
        return class_pred

# Returning the percentage of when the prediction matches the training samples.
def accuracy(y_pred, y_train):
    return np.sum(y_pred == y_train)/len(y_train)


if __name__=="__main__":
    log_class = logistic_classifer()
    log_class.SGD(X_train, y_train)                         # Training the dataset 


    y_pred = log_class.predict_test(X_train)                # Predicting the values of y_train, based on the training of X_train 
    
    acc = accuracy(y_pred, y_train)                         # Calculating the accuracy of the model
    print(f"The accuracy of the model is: {acc}\n\n")

    """2b"""
    print(f"\n\nProblem 2b\n")



    plt.ion()
    plt.figure()
    plt.scatter(X_train, y_pred)
    # plt.show(block=True)






"""2c: Bonus"""
# print(f"\n\nProblem 2c\n")