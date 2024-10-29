import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
# from Models.descision_tree import DecisionTreeClassifier

""" Load and preprocess the data """
def load_and_preprocess_data(train_path='train.csv', test_path='test.csv'):
    print("Loading data ...")
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)

    # Separating features and targets
    X = train_data.drop(['Id', 'lipophilicity'], axis=1)
    y = train_data['lipophilicity'].values
    X_test = test_data.drop(['Id'], axis=1)

    #Scaling the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_test_scaled = scaler.transform(X_test)

    return X_scaled, y, X_test_scaled, test_data['Id']

""" Performing cross-validation to find the best hyperparameters """
def perform_cross_validation(X, y, DecisionTreeClassifier):
    print("Cross validation on going ...")

    # Hyperparameters
    hyperparameters = {
        'max_depth': [3, 5, 7],                     # Tree depth
        'min_samples_splits': [5, 10],              # Minmum samples required to split a node
        'n_folds': 3,                               # Number of cross-validation folds
        'max_features': 'sqrt'                      # Number of features to consider at each split
    }

    best_params = {
        'depth': None,
        'min_samples': None,
        'score': -1
    }

    kf = KFold(n_splits=hyperparameters['n_folds'], shuffle=True, random_state=42)
    
    for depth in hyperparameters['max_depth']:
        for min_samples in hyperparameters['min_samples_splits']:
            scores = []
            print(f"Testing depth={depth}, min_samples_split={min_samples}")
            
            for train_idx, val_idx in kf.split(X):
                X_train_fold = X[train_idx]
                X_val_fold = X[val_idx]
                y_train_fold = y[train_idx]
                y_val_fold = y[val_idx]
                
                dt = DecisionTreeClassifier(
                    max_depth=depth,
                    min_samples_split=min_samples,
                    max_features=hyperparameters['max_features']
                )
                dt.fit(X_train_fold, y_train_fold)
                y_pred = dt.predict(X_val_fold)
                
                score = f1_score(y_val_fold, y_pred)
                scores.append(score)
            
            avg_score = np.mean(scores)
            print(f"Average F1 score: {avg_score:.4f}")
            
            if avg_score > best_params['score']:
                best_params['score'] = avg_score
                best_params['depth'] = depth
                best_params['min_samples'] = min_samples
    
    print(f"\nBest parameters found:")
    print(f"max_depth: {best_params['depth']}")
    print(f"min_samples_split: {best_params['min_samples']}")
    print(f"Best F1 score: {best_params['score']:.4f}")
    
    return best_params

""" Evaluation of the model and printing of the metrics """
def evaluate_model(model, X_val, y_val):
    y_val_pred = model.predict(X_val)

    print("\nModel Performance on Validation Set:")
    print(f"Accuracy: {accuracy_score(y_val, y_val_pred):.4f}")
    print(f"F1 score: {f1_score(y_val, y_val_pred):.4f}")
    print(f"Precision: {precision_score(y_val, y_val_pred):.4f}")
    print(f"Recall: {recall_score(y_val, y_val_pred):.4f}")

    print("\nDetailed Classification Report: ")
    print(classification_report(y_val, y_val_pred))

    return y_val_pred