import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report, precision_score, recall_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import KFold
# from Models.descision_tree import DecisionTreeClassifier

""" Load and preprocess the data """ 
def load_and_preprocess_data(scaling_method=None, train_path='train.csv', test_path='test.csv'):
    print("Loading data ...")
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)

    # Separating features and targets
    X = train_data.drop(['Id', 'lipophilicity'], axis=1)
    y = train_data['lipophilicity'].values
    X_test = test_data.drop(['Id'], axis=1)

    print("Preprocessing steps:")
    # Handle missing values if any
    print("- Handling missing values...")
    X = X.fillna(X.mean())
    X_test = X_test.fillna(X_test.mean())

    # Remove constant features
    print("- Removing constant features...")
    constant_features = [col for col in X.columns if X[col].std() == 0]
    if constant_features:
        print(f"  Removed {len(constant_features)} constant features")
        X = X.drop(columns=constant_features)
        X_test = X_test.drop(columns=constant_features)

    # Remove highly correlated features
    print("- Checking for highly correlated features...")
    correlation_matrix = X.corr().abs()
    upper = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))
    high_corr_features = [column for column in upper.columns if any(upper[column] > 0.95)]
    if high_corr_features:
        print(f"  Removed {len(high_corr_features)} highly correlated features")
        X = X.drop(columns=high_corr_features)
        X_test = X_test.drop(columns=high_corr_features)

    print(f"Final number of features: {X.shape[1]}")

    # Scale the features if method is specified
    if scaling_method:
        print(f"- Applying {scaling_method} scaling...")
        if scaling_method == 'standard':
            scaler = StandardScaler()
        elif scaling_method == 'robust':
            scaler = RobustScaler()
        else:
            raise ValueError("Scaling method must be either 'standard', 'robust', or None")
        
        X_processed = scaler.fit_transform(X)
        X_test_processed = scaler.transform(X_test)
    else:
        print("- Skipping scaling...")
        X_processed = X.values  # Convert to numpy array for consistency
        X_test_processed = X_test.values

    return X_processed, y, X_test_processed, test_data['Id']


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
        'max_depth': [7, 10, 15],  # Added deeper trees and None for unlimited depth
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
