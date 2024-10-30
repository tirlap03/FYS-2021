import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report, precision_score, recall_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import KFold
# from Models.descision_tree import DecisionTreeClassifier

""" Load and preprocess the data """
# def load_and_preprocess_data(scaling_method=None, train_path='train.csv', test_path='test.csv'):
#     print("Loading data ...")
#     train_data = pd.read_csv(train_path)
#     test_data = pd.read_csv(test_path)

#     # Separating features and targets
#     X = train_data.drop(['Id', 'lipophilicity'], axis=1)
#     y = train_data['lipophilicity'].values
#     X_test = test_data.drop(['Id'], axis=1)

#     # If no scaling requested, return unscaled data
#     if scaling_method is None:
#         return X, y, X_test, test_data['Id']

#     # Only call scaling if a method is specified
#     if scaling_method in ['standard', 'minmax', 'robust']:
#         X_scaled, X_test_scaled, _ = scaling(
#             X,
#             X_test,
#             method=scaling_method
#         )
#         return X_scaled, y, X_test_scaled, test_data['Id']
#     else:
#         raise ValueError(f"Unknown scaling method: {scaling_method}")
    
def load_and_preprocess_data(scaling_method='standard', train_path='train.csv', test_path='test.csv'):
    print("Loading data ...")
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)

    # Separating features and targets
    X = train_data.drop(['Id', 'lipophilicity'], axis=1)
    y = train_data['lipophilicity'].values
    X_test = test_data.drop(['Id'], axis=1)

    # Handle missing values if any
    X = X.fillna(X.mean())
    X_test = X_test.fillna(X_test.mean())

    # Remove constant features
    constant_features = [col for col in X.columns if X[col].std() == 0]
    X = X.drop(columns=constant_features)
    X_test = X_test.drop(columns=constant_features)

    # Remove highly correlated features
    correlation_matrix = X.corr().abs()
    upper = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))
    high_corr_features = [column for column in upper.columns if any(upper[column] > 0.95)]
    X = X.drop(columns=high_corr_features)
    X_test = X_test.drop(columns=high_corr_features)

    # Scale the features if method is specified
    if scaling_method:
        if scaling_method == 'standard':
            scaler = StandardScaler()
        elif scaling_method == 'robust':
            scaler = RobustScaler()
        
        X_scaled = scaler.fit_transform(X)
        X_test_scaled = scaler.transform(X_test)
        
        return X_scaled, y, X_test_scaled, test_data['Id']
    
    return X, y, X_test, test_data['Id']

""" Scaling features """
def scaling(X_train, X_test=None, method='standard'):
    if method is None:
        if X_test is not None:
            return X_train, X_test
        return X_train, None
    
    if method == 'standard':
        print("Standard scaling")
        scaler = StandardScaler()
    elif method == 'minmax':
        print("Minmax scaling")
        scaler = MinMaxScaler()
    elif method == 'robust':
        print("Robust scaling")
        scaler = RobustScaler()
    else: 
        raise ValueError(f"Unknown scaling method: {method}")
    
    X_train_scaled = scaler.fit_transform(X_train)

    if X_test is not None:
        X_test_scaled = scaler.transform(X_test)
        return X_train_scaled, X_test_scaled, scaler
    
    return X_train_scaled, scaler


""" Performing cross-validation to find the best hyperparameters """
def perform_cross_validation(X, y, DecisionTreeClassifier):
    print("Cross validation on going ...")

    # Hyperparameters
    hyperparameters = {
        'max_depth': [3, 5, 7, 11, 13],                     # Tree depth
        'min_samples_splits': [2, 5, 10, 15],              # Minmum samples required to split a node
        'n_folds': 5,                               # Number of cross-validation folds
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

""" Performing cross-validation to find the best hyperparameters for Random Forest """
def perform_cross_validation_rf(X, y, RandomForest):
    print("Performing Random Forest cross-validation...")
    
    if isinstance(X, pd.DataFrame):
        X = X.to_numpy()
    if isinstance(y, pd.DataFrame):
        y = y.to_numpy()

    # Expanded hyperparameters to try
    hyperparameters = {
        'n_trees': [100, 200, 300],  # Increased number of trees
        'max_depth': [7, 10, 15, None],  # Added deeper trees and None for unlimited depth
        'min_samples_split': [2, 5, 10],  # Added smaller min_samples_split
        'n_folds': 5  # Increased number of folds for more robust validation
    }
    
    best_params = {
        'n_trees': None,
        'depth': None,
        'min_samples': None,
        'score': -1
    }
    
    kf = KFold(n_splits=hyperparameters['n_folds'], shuffle=True, random_state=42)
    
    for n_trees in hyperparameters['n_trees']:
        for depth in hyperparameters['max_depth']:
            for min_samples in hyperparameters['min_samples_split']:
                scores = []
                print(f"Testing n_trees={n_trees}, depth={depth}, min_samples_split={min_samples}")
                
                for train_idx, val_idx in kf.split(X):
                    X_train_fold = X[train_idx]
                    X_val_fold = X[val_idx]
                    y_train_fold = y[train_idx]
                    y_val_fold = y[val_idx]
                    
                    rf = RandomForest(
                        n_trees=n_trees,
                        max_depth=depth,
                        min_samples_split=min_samples,
                        max_features='sqrt',
                        bootstrap=True  # Ensure bootstrapping is enabled
                    )
                    rf.fit(X_train_fold, y_train_fold)
                    y_pred = rf.predict(X_val_fold)
                    
                    score = f1_score(y_val_fold, y_pred)
                    scores.append(score)
                
                avg_score = np.mean(scores)
                print(f"Average F1 score: {avg_score:.4f}")
                
                if avg_score > best_params['score']:
                    best_params['score'] = avg_score
                    best_params['n_trees'] = n_trees
                    best_params['depth'] = depth
                    best_params['min_samples'] = min_samples
    
    print(f"\nBest parameters found:")
    print(f"n_trees: {best_params['n_trees']}")
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