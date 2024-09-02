import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE, SVMSMOTE, BorderlineSMOTE, ADASYN, RandomOverSampler, KMeansSMOTE
import joblib
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.svm import SVC

# Load data
# x_train = pd.read_csv('./data/yuge70_xtrain.csv')
# y_train = pd.read_csv('./data/yuge70_ytrain.csv')

x_train = pd.read_csv('./data/x_train70_bsmote.csv')
y_train = pd.read_csv('./data/y_train70_bsmote.csv')

# Drop unnecessary columns
smote = True
imba = False
print(x_train.columns)
if not smote:
    x_train = x_train.drop(['index'], axis=1)


# Ensure y_train is a Series and convert to the desired type
y_train = y_train['TOP_5'].astype(int)  # Convert to integer
# Alternatively, for boolean: y_train = y_train['TOP_5'].astype(bool)

# Check the unique values to ensure they are consistent
print(y_train.unique())

# Begin training
print(f'begin training for {x_train.columns}')
# model = RandomForestClassifier(
#     n_estimators=90,
#     max_depth=25,
#     min_samples_leaf=150,
#     class_weight='balanced',
#     max_features='log2',
#     min_samples_split=220,
#     bootstrap=False,
#     n_jobs=-1,
#     random_state=420
# )

model = RandomForestClassifier(
    n_estimators=120,
    max_depth=50,
    min_samples_leaf=200,
    max_features='sqrt',
    min_samples_split=200,
    bootstrap=False,
    n_jobs=-1,
    class_weight='balanced',
    random_state=420
)
if imba:

    # Applying ADASYN
    print("Applying ADASYN to the minority class")
    smote_model = adasyn = ADASYN(
        sampling_strategy=0.7,  # Adjust to balance the classes
        n_neighbors=5,          # Number of neighbors to use for generating samples
        random_state=42,        # Reproducibility
        n_jobs=-1               # Use all processors
    )
    x_train, y_train = smote_model.fit_resample(x_train, y_train)
    x_train.to_csv('./data/x_train70_adasyn.csv', index=False)
    y_train.to_csv('./data/y_train70_adasyn.csv', index=False)

    k_estimator = KMeans
    # Applying KMeans SMOTE
    print("Applying KMeans SMOTE to the minority class")
    smote_model = KMeansSMOTE(sampling_strategy=0.8,
                              kmeans_estimator=KMeans(n_clusters=10, random_state=420),
                              cluster_balance_threshold=0.2, k_neighbors=7,
                              random_state=420, n_jobs=-1)  # Use all CPUs
    x_train, y_train = smote_model.fit_resample(x_train, y_train)
    x_train.to_csv('./data/x_train70_kmeans.csv', index=False)
    y_train.to_csv('./data/y_train70_kmeans.csv', index=False)

    # Applying SVMSMOTE
    print("Applying SVM SMOTE to the minority class")

    # Configure the NearestNeighbors instance with n_jobs for parallel processing
    k_neighbors_estimator = NearestNeighbors(n_neighbors=5, n_jobs=-1)  # Use all CPUs

    # Initialize SVMSMOTE with the nearest neighbors estimator
    smote_model = SVMSMOTE(
        sampling_strategy=0.75,
        k_neighbors=k_neighbors_estimator,  # Pass the nearest neighbors estimator
        svm_estimator=SVC(kernel='rbf', C=1.0, gamma='scale'),  # Set SVM estimator
        random_state=420
    )

    x_train, y_train = smote_model.fit_resample(x_train, y_train)
    x_train.to_csv('./data/x_train70_svmsmote.csv', index=False)
    y_train.to_csv('./data/y_train70_svmsmote.csv', index=False)

    # Applying Borderline SMOTE
    print("Applying Borderline SMOTE to the minority class")
    smote_model = BorderlineSMOTE(sampling_strategy=0.7,
                                  random_state=420, k_neighbors=10,
                                  m_neighbors=10, kind='borderline-1')  # Use all CPUs
    x_train, y_train = smote_model.fit_resample(x_train, y_train)
    x_train.to_csv('./data/x_train70_bsmote.csv', index=False)
    y_train.to_csv('./data/y_train70_bsmote.csv', index=False)



# Fit the model
print('Fit has begun')
model.fit(x_train, y_train)
print('Fit completed')

# Save the model
joblib.dump(model, 'models/yuge70.joblib')
print('Model Saved')