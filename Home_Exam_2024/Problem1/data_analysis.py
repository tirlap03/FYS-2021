import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_dataset(train_path='train.csv', test_path='test.csv', save_plots=True):
    """
    Analyze the dataset and generate visualizations and statistics.
    
    Parameters:
        train_path (str): Path to training data CSV
        test_path (str): Path to test data CSV
        save_plots (bool): Whether to save plots to files instead of displaying
        
    Returns:
        dict: Dictionary containing key statistics about the dataset
    """
    # Load the datasets
    print("Loading datasets...")
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)
    
    # Calculate basic statistics
    n_samples_train = train_data.shape[0]
    n_samples_test = test_data.shape[0]
    n_features = train_data.shape[1] - 2  # Subtract Id and target columns
    
    print("\nBasic Dataset Information:")
    print(f"Training samples: {n_samples_train}")
    print(f"Test samples: {n_samples_test}")
    print(f"Number of features: {n_features}")
    
    # Target distribution
    target_dist = train_data['lipophilicity'].value_counts(normalize=True)
    print("\nTarget Distribution:")
    print(target_dist)
    
    # Check for missing values
    missing_train = train_data.isnull().sum()
    if missing_train.any():
        print("\nMissing values in training set:")
        print(missing_train[missing_train > 0])
    else:
        print("\nNo missing values in training set")
    
    # Feature statistics
    feature_data = train_data.drop(['Id', 'lipophilicity'], axis=1)
    feature_stats = feature_data.describe()
    print("\nFeature Statistics:")
    print(feature_stats)
    
    # Create visualizations
    print("\nGenerating visualizations...")
    
    # 1. Target Distribution Plot
    plt.figure(figsize=(10, 6))
    sns.countplot(data=train_data, x='lipophilicity')
    plt.title('Distribution of Lipophilicity Classes')
    plt.xlabel('Class')
    plt.ylabel('Count')
    if save_plots:
        plt.savefig('Data_analysis/target_distribution.png')
        plt.close()
    
    # 2. Feature Distributions (first 10 features)
    plt.figure(figsize=(15, 8))
    feature_data.iloc[:, :10].boxplot()
    plt.title('Distribution of First 10 Features')
    plt.xticks(rotation=45)
    plt.tight_layout()
    if save_plots:
        plt.savefig('Data_analysis/feature_distributions.png')
        plt.close()
    
    # 3. Correlation Matrix
    plt.figure(figsize=(12, 8))
    correlation_matrix = feature_data.corr()
    sns.heatmap(correlation_matrix, cmap='coolwarm', center=0)
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    if save_plots:
        plt.savefig('Data_analysis/correlation_matrix.png')
        plt.close()
    
    # 4. Feature importance based on correlation with target
    target_correlations = abs(feature_data.corrwith(train_data['lipophilicity']))
    top_features = target_correlations.nlargest(10)
    
    plt.figure(figsize=(12, 6))
    top_features.plot(kind='bar')
    plt.title('Top 10 Features by Correlation with Target')
    plt.xticks(rotation=45)
    plt.tight_layout()
    if save_plots:
        plt.savefig('Data_analysis/top_features.png')
        plt.close()
    
    # Collect statistics for return
    stats = {
        'n_samples_train': n_samples_train,
        'n_samples_test': n_samples_test,
        'n_features': n_features,
        'target_distribution': target_dist.to_dict(),
        'feature_ranges': {
            'min': feature_data.min().min(),
            'max': feature_data.max().max(),
            'mean': feature_data.mean().mean()
        },
        'top_correlated_features': top_features.to_dict(),
        'missing_values': missing_train[missing_train > 0].to_dict()
    }
    
    return stats

def main_da():
    """
    Run the data analysis and print a summary report
    """
    stats = analyze_dataset()
    
    # Print summary report
    print("\nDATASET ANALYSIS SUMMARY")
    print("=" * 50)
    print(f"Total training samples: {stats['n_samples_train']}")
    print(f"Total test samples: {stats['n_samples_test']}")
    print(f"Number of features: {stats['n_features']}")
    print("\nClass Distribution:")
    for class_label, proportion in stats['target_distribution'].items():
        print(f"Class {class_label}: {proportion:.2%}")
    
    print("\nFeature Value Ranges:")
    print(f"Min: {stats['feature_ranges']['min']:.2f}")
    print(f"Max: {stats['feature_ranges']['max']:.2f}")
    print(f"Mean: {stats['feature_ranges']['mean']:.2f}")
    
    print("\nTop 5 Most Correlated Features:")
    for feature, correlation in list(stats['top_correlated_features'].items())[:5]:
        print(f"{feature}: {correlation:.3f}")
