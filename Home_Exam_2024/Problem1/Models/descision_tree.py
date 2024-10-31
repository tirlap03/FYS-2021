import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from preprocessing import load_and_preprocess_data, perform_cross_validation
from utils import create_submission


class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

class DecisionTreeClassifier:
    def __init__(self, max_depth=None, min_samples_split=2, max_features='sqrt'):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.root = None

    def fit(self, X, y):
        self.n_classes = len(np.unique(y))
        self.n_features = X.shape[1]
        
        # Determine number of features to consider at each split
        if isinstance(self.max_features, str):
            if self.max_features == 'sqrt':
                self.n_features_split = int(np.sqrt(self.n_features))
            elif self.max_features == 'log2':
                self.n_features_split = int(np.log2(self.n_features))
        elif isinstance(self.max_features, float):
            self.n_features_split = int(self.max_features * self.n_features)
        else:
            self.n_features_split = self.n_features
            
        self.root = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        # Check for empty data
        if n_samples == 0:
            return Node(value=0)  # Return default class
            
        # Check if all samples belong to the same class
        if len(np.unique(y)) == 1:
            return Node(value=y[0])

        # Stopping criteria
        if (self.max_depth is not None and depth >= self.max_depth) or n_samples < self.min_samples_split or n_labels == 1:
            return Node(value=self._most_common_label(y))

        # Randomly select features to consider
        feature_indices = np.random.choice(
            n_features, 
            size=min(self.n_features_split, n_features), 
            replace=False
        )
        
        best_feature, best_threshold = self._best_split(X, y, feature_indices)
        
        if best_feature is None:
            return Node(value=self._most_common_label(y))

        # Create the split
        left_mask = X[:, best_feature] < best_threshold
        
        # Split the data
        X_left, y_left = X[left_mask], y[left_mask]
        X_right, y_right = X[~left_mask], y[~left_mask]
        
        # Create child nodes
        left = self._grow_tree(X_left, y_left, depth + 1)
        right = self._grow_tree(X_right, y_right, depth + 1)

        return Node(best_feature, best_threshold, left, right)

    def _best_split(self, X, y, feature_indices):
        best_gain = -1
        best_feature = None
        best_threshold = None

        for feature in feature_indices:
            # Get unique values for the feature, excluding extremes
            feature_values = X[:, feature]
            unique_values = np.unique(feature_values)

            if len(unique_values) < 2:
                continue

            thresholds = np.percentile(feature_values, [25, 50, 75])
            
            for threshold in thresholds:
                gain = self._information_gain(y, X[:, feature], threshold)

                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold

    def _information_gain(self, y, feature_values, threshold):
        parent_impurity = self._gini(y)

        # Create masks for left and right splits
        left_mask = feature_values < threshold
        
        # Get counts
        n = len(y)
        n_l = np.sum(left_mask)
        n_r = n - n_l
        
        if n_l == 0 or n_r == 0:
            return 0
        
        # Calculate child impurities
        child_impurity = (n_l / n) * self._gini(y[left_mask]) + \
                        (n_r / n) * self._gini(y[~left_mask])

        return parent_impurity - child_impurity

    def _gini(self, y):
        _, counts = np.unique(y, return_counts=True)
        proportions = counts / len(y)
        return 1 - np.sum(proportions ** 2)

    def _most_common_label(self, y):
        if len(y) == 0:
            return 0  # Return default class (0) if empty
        return np.bincount(y).argmax()

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, x, node):
        if node.value is not None:
            return node.value

        if x[node.feature] < node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)


def main_dt():
    # Firstly, load and process data
    X, y, X_test, test_ids = load_and_preprocess_data(None)

    if isinstance(X, pd.DataFrame):
        X = X.to_numpy()
    if isinstance(X_test, pd.DataFrame):
        X_test = X_test.to_numpy()

    # Secondly, finding the best hyperparameters through cross-validation
    best_params = perform_cross_validation(X, y, DecisionTreeClassifier)

    # Thirdly, Train and evaluate the model on validation set
    print("\nTraining and avlauating model ...")
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    model = DecisionTreeClassifier(
        max_depth=best_params['depth'],
        min_samples_split=best_params['min_samples'],
        max_features='sqrt'
    )
    model.fit(X_train, y_train)

    print("\nTraining final model on full dataset ...")
    final_model = DecisionTreeClassifier(
        max_depth=best_params['depth'],
        min_samples_split=best_params['min_samples'],
        max_features='sqrt'
    )
    final_model.fit(X, y)

    test_predictions = final_model.predict(X_test)

    create_submission(test_predictions, test_ids)

