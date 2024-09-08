import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation 
import seaborn as sns

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
print(f"X_train: 80% of the train set \n{X_train}\nWith the shape {X_train.shape}\n\n X_test: 20% of the train set \n{X_test}\nWith the shape of {X_test.shape}\n\n")
print(f"y_train: 80% of the test set \n{y_train}\nWith the shape of {y_train.shape}\n\n y_test:20% of the test set \n{y_test}\nWith the shape of {y_test.shape}\n\n")


"""1d: Bonus"""
# print(f"\n\nProblem 1d\n")

""""""""""""""""""""""""""""""""""""""""""""""""""""""

"""2a"""
print(f"\n\nProblem 2a\n")
training_errors = [] 
class logistic_classifer ():
    def __init__(self, lr=0.01, epochs=100, bias=None):
        self.lr = lr                    # learning rate
        self.epochs = epochs            # epochs = 100    
        self.weights = None             
        self.bias = None   
                  

    def zigmoid(self, lin_reg):
        lin_reg = np.clip(lin_reg, -500, 500)       # Clipping the values below and over 500, to avoid overflow
        z = 1/(1+np.exp(-lin_reg))                  # Calculating the sigmoid function    
        return z    
    

    def loss_func(self, pred_y, true_y):
        L = true_y*np.log10(pred_y) + (1-true_y)*np.log10(1-pred_y)
        # print(f"The loss is: {L.shape}, the type is {type(L)}\n")
        return L.mean()
    
    
    def SGD(self, X_train, y_train):
        self.bias = 0                               # Initializing the bias as 0
        samples, features = X_train.shape           # Initializing the number of samples and features as the rows and columns of the matrix
        self.weights = np.zeros(features)           # Initializing the weights as 0 (as many as there are features, here there is two)
        
        for epoch in range(self.epochs):
            if epoch % 10 == 0:
                print(f"You are on epoch number {epoch}\n")
            for i in range(len(X_train)):             
                lin_pred = np.dot(X_train, self.weights) + self.bias        # Calculating the predictions of the linear regression
                predictions = self.zigmoid(lin_pred)                        # Calculating the actual predictions, which is the sigmoid function

                # Calculating the gradient
                dw = (1/samples) * np.dot(X_train.T, (predictions - y_train))       
                db = (1/samples) * np.sum(predictions-y_train)

                # Calculating gradient, and updating the weights
                self.weights = self.weights - self.lr *dw
                self.bias = self.bias - self.lr*db
                
            loss = self.loss_func(predictions, lin_pred)
            training_errors.append(np.sum(loss) / samples)
                           
    # Predicting the values of y_train based on the training of X_train
    def predict_test(self, X_train):
        linear_pred = np.dot(X_train, self.weights) + self.bias
        # print(f"linreg: {lin_reg}\n\n")
        y_pred = self.zigmoid(linear_pred)

        class_pred = [0 if y<=0.5 else 1 for y in y_pred]
        return class_pred

# Plotting the training error as a function of epochs
def training_errorVSepochs(epochs, training_errors):
    # plt.plot(np.arange(epochs), training_errors)
    # plt.xlabel('Epochs')
    # plt.ylabel('Average Training Error')
    # plt.title('Training Error vs. Epochs')
    # plt.grid(True)

    plt.plot(range(epochs), training_errors, label='Training Error')
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.title('Training Error per Epoch')
    plt.legend()

    plt.savefig('traning_errorVSEpochs.png')
    plt.show(block=True)

def sigmoid_3D_plot(log_class):
    # Generate a grid of values for two features
    x = np.linspace(-10, 10, 100)
    y = np.linspace(-10, 10, 100)
    X, Y = np.meshgrid(x, y)

    # Compute the linear combination z = w1*x + w2*y + b using the learned weights and bias
    Z = log_class.weights[0] * X + log_class.weights[1] * Y + log_class.bias
    
    # Apply the sigmoid function to the linear combination
    Z_sig = log_class.zigmoid(Z)
    
    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the surface
    ax.plot_surface(X, Y, Z_sig, cmap='viridis')

    # Set labels and title
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_zlabel('Sigmoid Output')
    ax.set_title('3D Sigmoid Function (S-curve)')
    plt.savefig('3D_sigmoid.png')
    plt.show(block=True)



# Returning the percentage of when the prediction matches the training samples.
def accuracy(y_pred, y_train):
    return np.sum(y_pred == y_train)/len(y_train)

def plot_confusion_matrix(cm):
    plt.figure(figsize=(6,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('False Positive      True Negative')
    plt.ylabel('Confusion Matrix')
    plt.title('True Positive        False Negative')
    plt.savefig('confusion_matrix.png')
    plt.show(block=True)

def confusion_matrix_accuracy(cm):
    # Extract values from the confusion matrix
    TN, FP, FN, TP = cm.ravel()
    
    # Calculate accuracy
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    return accuracy


if __name__=="__main__":
    log_class = logistic_classifer()
    log_class.SGD(X_train, y_train)                         # Training the dataset 


    y_train_pred = log_class.predict_test(X_train)                # Predicting the values of y_train, based on the training of X_train 
    
    train_acc = accuracy(y_train_pred, y_train)                         # Calculating the accuracy of the model
    print(f"The accuracy of the model with the training set is: {train_acc}\n\n")

    """2b"""
    print(f"\n\nProblem 2b\n")

    y_test_pred = log_class.predict_test(X_test)
    test_acc = accuracy(y_test_pred, y_test)
    print(f"The accuracy of the model with the test set is: {test_acc}\n\n")

    
    training_errorVSepochs(log_class.epochs, training_errors)
    # sigmoid_3D_plot(log_class)

    
    """2c: Bonus"""
    # print(f"\n\nProblem 2c\n")


    """3a"""
    print(f"\n\nProblem 3a\n")

    cm_te = confusion_matrix(y_test, y_test_pred)
    print(f"Confusion matrix of the test set: \n{cm_te}\n")

    cm_te_acc = confusion_matrix_accuracy(cm_te)
    print(f"The accuracy of the test_set confusion matrix is {cm_te_acc}\n")

    plot_confusion_matrix(cm_te)