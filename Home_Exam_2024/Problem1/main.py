from Models.descision_tree import main_dt
from Models.logistic_regression import main_lr
from Models.random_forest import main_rf_cross, main_rf_no_cross
from data_analysis import analyze_dataset

def main():
    print("Analyzing dataset ...")
    stats = analyze_dataset()

    print("Running Decision Tree Classifier ...")
    main_dt()

    print("Running Logistic Regression Classifier ...")
    main_lr()

    print("Running Random Forest Classifier without cross validation...")
    main_rf_no_cross()

    print("Running Random Forest Classifier with cross validation ...")
    main_rf_cross()

if __name__ == "__main__":
    main()