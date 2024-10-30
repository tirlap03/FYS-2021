import numpy as np
from sklearn.model_selection import train_test_split

from preprocessing import load_and_preprocess_data, evaluate_model
from utils import create_submission


class LogisticClassifier ():
    def __init__(self, lr=0.01, epochs=100, bias=None):
        self.lr = lr                    # learning rate
        self.epochs = epochs            # epochs = 100    
        self.weights = None             
        self.bias = None   

    def sigmoid(self, lin_reg):
        lin_reg = np.clip(lin_reg, -500, 500)       # Clipping the values below and over 500, to avoid overflow
        z = 1/(1+np.exp(-lin_reg))                  # Calculating the sigmoid function    
        return z    

    def loss_func(self, pred_y, true_y):
        epsilon = 1e-15
        pred_y = np.clip(pred_y, epsilon, 1 - epsilon)
        L = true_y*np.log(pred_y) + (1-true_y)*np.log(1-pred_y)
        return L.mean()
    
    def SGD(self, X_train, y_train):
        training_errors = []
        self.bias = 0                               # Initializing the bias as 0
        samples, features = X_train.shape           # Initializing the number of samples and features as the rows and columns of the matrix
        self.weights = np.zeros(features)           # Initializing the weights as 0 (as many as there are features, here there is two)
        
        for epoch in range(self.epochs):
            lin_pred = np.dot(X_train, self.weights) + self.bias        # Calculating the predictions of the linear regression
            predictions = self.sigmoid(lin_pred)                        # Calculating the actual predictions, which is the sigmoid function

            # Calculating the gradient
            dw = (1/samples) * np.dot(X_train.T, (predictions - y_train))       
            db = (1/samples) * np.sum(predictions-y_train)

            # Calculating gradient, and updating the weights
            self.weights = self.weights - self.lr *dw
            self.bias = self.bias - self.lr*db
                
            loss = self.loss_func(predictions, lin_pred)
            training_errors.append(np.sum(loss) / samples)
                            
    # Predicting the values of y_train based on the training of X_train
    def predict(self, X):
        linear_pred = np.dot(X, self.weights) + self.bias
        y_pred = self.sigmoid(linear_pred)

        class_pred = (y_pred >= 0.5).astype(int)
        return class_pred
    

def hyperparameter_tuning(X, y):
    print("Tuning the hyperparameters of Logistic Regression ...")


    learning_rates = [0.0001, 0.001, 0.01, 0.1, 0.5]
    epochs = [10000, 50000, 100000]

    best_params = {
        'lr': None,
        'epochs': None,
        'score': -float('inf')
    }

    X_train, X_test, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    for lr in learning_rates:
        for epoch in epochs:
            print(f"Testing lr={lr}, epochs={epoch}")
            
            model = LogisticClassifier(lr=lr, epochs=epoch)
            model.SGD(X_train, y_train)
            
            # Evaluate on validation set
            y_pred = model.predict(X_test)
            score = np.mean(y_pred == y_val)
            
            if score > best_params['score']:
                best_params['score'] = score
                best_params['lr'] = lr
                best_params['epochs'] = epoch
    
    print(f"\nBest parameters found:")
    print(f"Learning rate: {best_params['lr']}")
    print(f"Epochs: {best_params['epochs']}")
    print(f"Best accuracy score: {best_params['score']:.4f}")
    
    return best_params


def main_lr():
    X_scaled, y, X_test_scaled, test_ids = load_and_preprocess_data('robust')

    best_params = hyperparameter_tuning(X_scaled, y)

    print("Training and evaluating model ...")
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    model = LogisticClassifier(
        lr=best_params['lr'],
        epochs=best_params['epochs']
    )
    model.SGD(X_train, y_train)

    y_val_pred = model.predict(X_test)
    evaluate_model(model, X_test, y_test)

    print("\nTraining the final model ...")
    final_model = LogisticClassifier(
        lr=best_params['lr'],
        epochs=best_params['epochs']
    )

    final_model.SGD(X_scaled, y)

    test_predictions = final_model.predict(X_test_scaled)

    create_submission(test_predictions, test_ids)