import pandas as pd


def create_submission(test_predictions, test_ids, filename='submission5.csv'):
    try:
        submission = pd.DataFrame({
            'Id': test_ids,
            'lipophilicity': test_predictions
        })
        
        submission.to_csv(filename, index=False)
        print(f"Submission file created: {filename}")
        
    except Exception as e:
        print(f"Error creating submission file: {e}")
