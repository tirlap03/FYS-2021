import pandas as pd
import os


def create_submission(test_predictions, test_ids, filename='submission_rf_cross2.csv'):
    try:
        filepath = os.path.join('Results', filename)

        submission = pd.DataFrame({
            'Id': test_ids,
            'lipophilicity': test_predictions
        })
        
        submission.to_csv(filepath, index=False)
        print(f"Submission file created: {filepath}")
        
    except Exception as e:
        print(f"Error creating submission file: {e}")
