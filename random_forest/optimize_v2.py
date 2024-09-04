import pandas as pd
import joblib
from skopt import BayesSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, precision_score, f1_score
from sklearn.model_selection import StratifiedKFold

# Load the training and validation datasets
x_train = pd.read_csv('./v2_top4/train_fold_11.csv', index_col='Dates')
x_valid = pd.read_csv('./v2_top4/test_fold_11.csv', index_col='Dates')

# Columns to drop
noninformative_columns = ['Ticker', 'set', 'window']
informative_columns = ['DELTA_QUANTILE', 'DELTA_20']

# Prepare the training dataset
y_train = x_train['TOP_QUANTILE']  # Extract the target variable
x_train = x_train.drop(noninformative_columns + informative_columns + ['PCT_CHANGE_20', 'TOP_QUANTILE'], axis=1)

# Prepare the validation dataset
y_valid = x_valid['TOP_QUANTILE']  # Extract the target variable
x_valid = x_valid.drop(noninformative_columns + informative_columns + ['PCT_CHANGE_20', 'TOP_QUANTILE'], axis=1)

# Store data into a dictionary
datadict = dict(x_train=x_train, y_train=y_train, x_valid=x_valid, y_valid=y_valid)

# Initialize the RandomForestClassifier
rf = RandomForestClassifier(random_state=420, n_jobs=-1)

# Define the custom scorer to focus on precision for the positive class (label 1)
precision_scorer = make_scorer(f1_score, average='weighted')  # Optimize for precision of all classes

# Define the search space for Bayesian optimization
search_space = {
    'n_estimators': (100, 1000),  # Range of number of trees
    'max_depth': (10, 100),  # Range of maximum depth of trees
    'criterion': ['gini', 'entropy', 'log_loss'],
    'min_samples_split': (2, 100),  # Minimum number of samples to split a node
    'min_samples_leaf': (1, 100),  # Minimum number of samples at a leaf node
    'max_features': ['sqrt', 'log2'],  # Number of features for the best split
    'min_weight_fraction_leaf': (0., 0.5),
    'class_weight': ['balanced', 'balanced_subsample'],  # Handle imbalance
    'bootstrap': [True, False]  # Whether bootstrap samples are used when building trees
}

# Perform BayesSearchCV with the custom precision scorer
bayes_search = BayesSearchCV(
    estimator=rf,
    search_spaces=search_space,
    scoring=precision_scorer,  # Use the precision scorer
    cv=StratifiedKFold(n_splits=5),
    n_iter=25,  # Number of iterations
    n_jobs=-1,
    random_state=42
)

# Fit the BayesSearchCV
print('Begin Bayesian fit...')
bayes_search.fit(x_train, y_train)

# Get the best model, parameters, and scores
best_model = bayes_search.best_estimator_
best_params = bayes_search.best_params_
best_precision = bayes_search.best_score_

print("Best Parameters:", best_params)
print("Best Precision Score:", best_precision)

# Save the best model, parameters, and data dictionary
joblib.dump(best_model, 'bestmodel3.joblib')
joblib.dump(bayes_search, 'bayes.joblib')
joblib.dump(best_params, 'bestparams.joblib')
joblib.dump(best_precision, 'bestprecision.joblib')
joblib.dump(datadict, 'datadict.joblib')