import os
import ydf
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score
import numpy as np

def main():
    train_data = pd.read_csv(f'./data/bigtrain.csv')
    valid_data = pd.read_csv(f'./data/bigvalid.csv')

    train_data.drop(['Ticker', 'PCT_CHANGE_20', 'index'], axis=1, inplace=True)

    import numpy as np
    label = 'TOP_5'

    # learner = ydf.RandomForestLearner(task=ydf.Task.CLASSIFICATION, label=label, num_trees=10000,
    #                                   winner_take_all=False, growing_strategy='BEST_FIRST_GLOBAL').train(train_data)

    learner = (ydf.RandomForestLearner(task=ydf.Task.CLASSIFICATION,
                                       label='TOP_5',
                                       max_depth = 100,
                                       # growing_strategy='BEST_FIRST_GLOBAL',
                                       num_trees=1000).train(train_data))

    valid_preds = learner.predict(valid_data.drop('TOP_5', axis=1, inplace=False))
    # Assuming valid_preds contains the probabilities of class 1
    threshold = 0.5
    predicted_classes = (valid_preds >= threshold).astype(int)

    # Now create a DataFrame for the predictions
    preds = pd.DataFrame({
        'Predicted': predicted_classes,
        'Probs': valid_preds
    })

    # Ensure consistency in lengths and alignment
    true_classes = valid_data["TOP_5"].reset_index(drop=True)
    predicted_classes = preds['Predicted'].reset_index(drop=True)

    # Check if lengths match
    assert len(true_classes) == len(predicted_classes), "Lengths of true and predicted classes do not match."

    # Calculate accuracies
    total_accuracy = np.mean(true_classes == predicted_classes)

    print("Accuracy: ", total_accuracy)

    model_path = "./models/big_29Aug2024"

    learner.to_tensorflow_saved_model(path=model_path)

    print(f'Model saved to path: {model_path}')

if __name__ == '__main__':
    main()