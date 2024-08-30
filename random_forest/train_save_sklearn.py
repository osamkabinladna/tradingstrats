import numpy as np
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

train_data = pd.read_csv(f'./data/bigtrain.csv')
valid_data = pd.read_csv(f'./data/bigvalid.csv')

y_train = train_data[['TOP_5']].values.ravel()
x_train = train_data.drop(['TOP_5', 'index', 'Ticker', 'DELTA_20_QUINTILES'], axis=1)

x_valid = valid_data.drop(['index', 'Ticker'], axis=1)
y_valid = valid_data[['TOP_5']].values.ravel()

print(f'begin training for {x_train.columns}')
model = RandomForestClassifier(n_estimators=300, max_depth=35,
                               min_samples_leaf=10, class_weight='balanced',
                               min_samples_split=25,n_jobs=-1,
                               random_state=420)
model.fit(x_train, y_train)
print('Fit completed')
joblib.dump(model, 'models/400tree35depth300tickers.joblib')
print('Model Saved')