import numpy as np
import pandas as pd
from Models.descision_tree import DecisionTreeClassifier
from utils import create_submission
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score, f1_score, classification_report, precision_score, recall_score
from preprocessing import load_and_preprocess_data, perform_cross_validation_rf

class RandomForestClassifier:
    def __init__(self, n_trees=100, max_depth=None, min_samples_split=2, max_features='sqrt', bootstrap=True):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.trees = []

    def fit(self, X, y):
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
        if isinstance(y, pd.DataFrame):
            y = y.to_numpy()

        # Calculate class weights
        unique_classes, class_counts = np.unique(y, return_counts=True)
        class_weights = dict(zip(unique_classes, 1 / class_counts))
        
        self.trees = []
        n_samples = X.shape[0]
        
        for _ in range(self.n_trees):
            # Balanced bootstrap sampling
            if self.bootstrap:
                # Calculate sampling weights based on class weights
                sample_weights = np.array([class_weights[yi] for yi in y])
                sample_weights = sample_weights / sample_weights.sum()
                
                indices = np.random.choice(
                    n_samples, 
                    size=n_samples, 
                    replace=True,
                    p=sample_weights
                )
                X_bootstrap = X[indices]
                y_bootstrap = y[indices]
            else:
                X_bootstrap = X
                y_bootstrap = y
            
            tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                max_features=self.max_features
            )
            tree.fit(X_bootstrap, y_bootstrap)
            self.trees.append(tree)

    def predict(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()

        # Get predictions from all trees
        tree_predictions = np.array([tree.predict(X) for tree in self.trees])
        
        # Take weighted majority vote
        return np.array([
            np.bincount(predictions, minlength=2).argmax()
            for predictions in tree_predictions.T
        ])



def evaluate_model(model, X_val, y_val):
    y_val_pred = model.predict(X_val)
    
    print("\nModel Performance on Validation Set:")
    print(f"Accuracy: {accuracy_score(y_val, y_val_pred):.4f}")
    print(f"F1 score: {f1_score(y_val, y_val_pred):.4f}")
    print(f"Precision: {precision_score(y_val, y_val_pred):.4f}")
    print(f"Recall: {recall_score(y_val, y_val_pred):.4f}")
    
    print("\nDetailed Classification Report:")
    print(classification_report(y_val, y_val_pred))
    
    return y_val_pred

def main_rf_cross():
    # Load and process data
    X_scaled, y, X_test_scaled, test_ids = load_and_preprocess_data(scaling_method=None)
    
    # Find best hyperparameters through cross-validation
    best_params = perform_cross_validation_rf(X_scaled, y, RandomForestClassifier)
    
    # Train and evaluate model on validation set
    print("\nTraining and evaluating model...")
    X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(
        n_trees=best_params['n_trees'],
        max_depth=best_params['depth'],
        min_samples_split=best_params['min_samples'],
        max_features='sqrt'
    )
    model.fit(X_train, y_train)
    
    y_val_pred = evaluate_model(model, X_val, y_val)
    
    # Train final model on full dataset
    print("\nTraining final model on full dataset...")
    final_model = RandomForestClassifier(
        n_trees=best_params['n_trees'],
        max_depth=best_params['depth'],
        min_samples_split=best_params['min_samples'],
        max_features='sqrt'
    )
    final_model.fit(X_scaled, y)
    
    test_predictions = final_model.predict(X_test_scaled)
    
    # Create submission file
    create_submission(test_predictions, test_ids)

def main_rf_no_cross():
    # Load and process data
    X_scaled, y, X_test_scaled, test_ids = load_and_preprocess_data(scaling_method=None)
    
    # Split data into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    # Create and train model with default parameters
    model = RandomForestClassifier(
        n_trees=100,
        max_depth=5,
        min_samples_split=5,
        max_features='sqrt'
    )
    model.fit(X_train, y_train)
    
    # Evaluate model
    y_val_pred = evaluate_model(model, X_val, y_val)
    
    # Train final model on full dataset
    print("\nTraining final model on full dataset...")
    final_model = RandomForestClassifier(
        n_trees=100,
        max_depth=5,
        min_samples_split=5,
        max_features='sqrt'
    )
    final_model.fit(X_scaled, y)
    
    test_predictions = final_model.predict(X_test_scaled)
    
    # Create submission file
    create_submission(test_predictions, test_ids)

# if __name__ == "__main__":
#     # Choose which main function to run
#     use_cv = True  # Set to False to run without cross-validation
    
#     if use_cv:
#         main_rf_with_cv()
#     else:
#         main_rf_simple()