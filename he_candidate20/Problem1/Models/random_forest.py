import numpy as np
import pandas as pd
from Models.descision_tree import DecisionTreeClassifier
from preprocessing import load_and_preprocess_data, evaluate_model, perform_cross_validation_rf
from utils import create_submission
from sklearn.model_selection import train_test_split
# from imblearn.over_sampling import SMOTE

class RandomForest:
    def __init__(self, n_trees=100, max_depth=None, min_samples_split=2, max_features='sqrt', class_weights=None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.trees = []
        self.class_weights = class_weights

    """ Building a forest of trees from the training set (X, y) """
    def fit(self, X, y):
        """
        Parameters:
            X: array-like of shape (n_samples, n_features)
                - Training data
            y: array-like of shape (n_samples, )
                - Target values 
        Returns:
            self: object
                - Returns self (the fitted model)
        """
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
        if isinstance(y, pd.Series):
            y = y.to_numpy()
            
        # Validate input
        if len(X) == 0 or len(y) == 0:
            raise ValueError("Empty dataset provided")
        
        if len(np.unique(y)) < 2:
            raise ValueError("Dataset contains only one class")
                
        # Calculate class weights if not provided
        if self.class_weights == 'balanced':
            unique_classes, class_counts = np.unique(y, return_counts=True)
            total_samples = len(y)
            self.class_weights = {
                cls: total_samples / (len(unique_classes) * count)
                for cls, count in zip(unique_classes, class_counts)
            }

        for _ in range(self.n_trees):
            X_sample, y_sample = self._bootstrap_sample(X, y)
            
            # Validate bootstrap sample
            if len(np.unique(y_sample)) < 2:
                continue  # Skip this tree if sample has only one class
                
            tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                max_features=self.max_features
            )
            
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)
            
        if len(self.trees) == 0:
            raise ValueError("Could not create any valid trees")

        return self

    """ Make predictions using the forest. Predict a class for X """
    def predict(self, X):
        """
        Parameters:
            X: array-like of shape (n-samples, n_features)
                - The input samples to predict
        Returns:
            y_pred: array-like f shape (n_samples, )
                - The predicted classes
        """
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
        # Get predictions from all trees
        tree_predictions = np.array([tree.predict(X) for tree in self.trees])

        # tree_predictions shape is (n_trees, n_samples)
        # Taking the majority vote for each sample
        return self._majority_vote(tree_predictions)

    """ Create a bootstrap dataset for each tree """
    def _bootstrap_sample(self, X, y):
        """
        Parameters:
            X: array-like of shape (n_samples, n_features)
                - Training data
            y: array-like of shape (n_sample, )
                - Target values
        Returns:
            X_sample: array-like of shape (n_samples, n_features)
                - Bootstrapped training data
            y_sample: array-like of shape (n_samples, )
                - Bootstrapped target values
        """
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
        
        n_samples = X.shape[0]
        
        # Get indices for each class
        class_0_idx = np.where(y == 0)[0]
        class_1_idx = np.where(y == 1)[0]
        
        # Ensure we have at least one sample from each class
        if len(class_0_idx) == 0 or len(class_1_idx) == 0:
            # If one class is empty, use original distribution
            sample_idx = np.random.choice(len(y), size=n_samples, replace=True)
            return X[sample_idx], y[sample_idx]
        
        # Get the size of majority class
        max_size = max(len(class_0_idx), len(class_1_idx))
        
        # Resample minority class to match majority class
        if len(class_0_idx) < len(class_1_idx):
            class_0_idx = np.random.choice(class_0_idx, size=max_size, replace=True)
        else:
            class_1_idx = np.random.choice(class_1_idx, size=max_size, replace=True)
        
        # Combine indices and shuffle
        combined_idx = np.concatenate([class_0_idx, class_1_idx])
        np.random.shuffle(combined_idx)
        
        # Create balanced bootstrap sample
        sample_idx = np.random.choice(combined_idx, size=n_samples, replace=True)
        
        return X[sample_idx], y[sample_idx]

    """ Combine predictions from all trees """
    def _majority_vote(self, predictions):
        """
        Parameters:
            predictions: array-like of shape (n_trees, n_samples)
                - Predictions from all trees for all samples
        
        Returns:
            array-like of shape (n_samples, )
                - Majority vote prediction for each sample
        """
        # For each sample, count which class (0 or 1) appears more often
        return np.round(np.mean(predictions, axis=0).astype(int))

def main_rf_no_cross():
    # Loading and preprocess data
    X_scaled, y, X_test_scaled, test_ids = load_and_preprocess_data(scaling_method=None)       # No scaling

    # Split data for validation
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

    print("\nTraining Random Forest ...")
    rf_model = RandomForest(
        n_trees=100,                # Number of trees in the forest
        max_depth=5,                # maximum depth of each tree
        min_samples_split=5,        # Minimum samples required to split a node
        max_features='sqrt',        # Number of features to consider for the split
        class_weights='balanced'
    )
    rf_model.fit(X_train, y_train)


    # Train final model on full dataset
    print("\nTraining final model on full dataset ...")
    final_rf_model = RandomForest(
        n_trees=100,                # Number of trees in the forest
        max_depth=5,                # maximum depth of each tree
        min_samples_split=5,        # Minimum samples required to split a node
        max_features='sqrt',        # Number of features to consider for the split
        class_weights='balanced'
    )
    final_rf_model.fit(X_scaled, y)

    print("\nMaking predictions on test set ...")
    test_predictions = final_rf_model.predict(X_test_scaled)

    create_submission(test_predictions, test_ids)

def main_rf_cross():
    # Load and preprocess data
    print("Loading and preprocessing data...")
    X_scaled, y, X_test_scaled, test_ids = load_and_preprocess_data(scaling_method=None)
    
    # Find best hyperparameters through cross-validation
    print("\nPerforming cross-validation...")
    best_params = perform_cross_validation_rf(X_scaled, y, RandomForest)
    
    # Split data for final validation
    print("\nSplitting data for validation...")
    X_train, X_val, y_train, y_val = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train and evaluate model with best parameters
    print("\nTraining and evaluating model...")
    model = RandomForest(
        n_trees=best_params['n_trees'],
        max_depth=best_params['depth'],
        min_samples_split=best_params['min_samples'],
        max_features=best_params['max_features'],
        class_weights='balanced'
    )
    model.fit(X_train, y_train)
    
    # Train final model on full dataset
    print("\nTraining final model on full dataset...")
    final_model = RandomForest(
        n_trees=best_params['n_trees'],
        max_depth=best_params['depth'],
        min_samples_split=best_params['min_samples'],
        max_features=best_params['max_features'],
        class_weights='balanced'
    )
    final_model.fit(X_scaled, y)
    
    # Make predictions and create submission
    test_predictions = final_model.predict(X_test_scaled)
    create_submission(test_predictions, test_ids)
