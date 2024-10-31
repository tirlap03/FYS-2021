# Machine Learning Home Exam (FYS-2021)

This repository contains the implementation of two machine learning problems for the FYS-2021 Machine Learning course home exam.

## Project Structure

```
├── Doc/
│   └── report.pdf
├── Problem1/
│   ├── main.py
│   ├── preprocessing.py
│   ├── utils.py
│   ├── data_analysis.py
│   └── Models/
│       └── decision_tree.py
|       └── logistic_regression.py
|       └── random_forest.py
├── Problem2/
│   └── frey-faces.csv
│   └── KMeans.py
│   └── Problem2.py
│   └── Results/
└── README.md
```

## Requirements

The project requires the following Python packages:
- numpy
- pandas
- scikit-learn (for Problem 1)
- matplotlib
- seaborn

You can install them using:
```bash
pip install numpy pandas scikit-learn matplotlib seaborn
```

## Problem 1: Chemical Data Classification

This problem involves classifying chemical data using various machine learning methods. The implementation includes a custom decision tree classifier and support for other scikit-learn methods.

### Data
- Download the dataset from the Kaggle competition link provided in the exam
- Place the `train.csv` and `test.csv` files in the `Problem1` directory

### Running Problem 1
```in your terminal
cd Problem1
python main.py
```

This will:
1. Load and preprocess the data
2. Perform cross-validation to find optimal hyperparameters
3. Train the model and evaluate on a validation set
4. Generate predictions for the test set
5. Create a submission file (`submission.csv`)

## Problem 2: Frey Faces Clustering

This problem implements k-means clustering on the Frey faces dataset.

### Data
- Place the `frey_faces.csv` file in the `Problem2` directory

### Running Problem 2
``` in your terminal
cd Problem2
python main.py
```

This will:
1. Load the Frey faces dataset
2. Apply k-means clustering
3. Generate visualizations of the clusters

## Report

The complete report for both problems can be found in `Doc/report.pdf`
