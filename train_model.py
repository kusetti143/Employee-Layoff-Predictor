import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import xgboost as xgb
import joblib
import os
import glob
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

# Set the training dataset file path here
TRAIN_FILE = 'TrainDatasets/Emp_Train_2.csv'  # Change this to use a different dataset

# 1. Load Data
data = pd.read_csv(TRAIN_FILE)

# Debug: Show data info and head
print('--- Data Info ---')
print(data.info())
print('--- Data Head ---')
print(data.head())

# Debug: Check class balance
print('--- Target Value Counts (Layoff) ---')
print(data['Layoff'].value_counts())
if data['Layoff'].value_counts(normalize=True).min() < 0.2:
    print('WARNING: Class imbalance detected! Consider using class_weight or resampling techniques.')

# 2. Preprocessing
# Dynamically detect categorical columns (object type, except EmployeeID and Layoff)
categorical_cols = [col for col in data.columns if data[col].dtype == 'object' and col not in ['EmployeeID', 'Layoff']]
# One-hot encode categorical variables
data = pd.get_dummies(data, columns=categorical_cols, drop_first=True)

# Features and target
X = data.drop(['EmployeeID', 'Layoff'], axis=1)
y = data['Layoff']

# Debug: Show feature columns after encoding
print('--- Feature Columns After Encoding ---')
print(list(X.columns))

# Dynamically detect numeric columns for scaling (exclude one-hot columns)
numeric_cols = [col for col in ['Tenure', 'Salary', 'PerformanceScore'] if col in X.columns]
scaler = StandardScaler()
if numeric_cols:
    X[numeric_cols] = scaler.fit_transform(X[numeric_cols])
    print('--- Scaled Features Sample ---')
    print(X[numeric_cols].head())
else:
    print('No numeric columns to scale.')

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Apply SMOTE to training data
print('Before SMOTE:', pd.Series(y_train).value_counts())
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
print('After SMOTE:', pd.Series(y_train_res).value_counts())

# 3. Model Training
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
    return {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_proba) if y_proba is not None else None
    }

results = {}
models = {}

# Logistic Regression with class_weight
lr = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
lr.fit(X_train_res, y_train_res)
results['Logistic Regression'] = evaluate_model(lr, X_test, y_test)
models['Logistic Regression'] = lr

# Random Forest with class_weight
rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
rf.fit(X_train_res, y_train_res)
results['Random Forest'] = evaluate_model(rf, X_test, y_test)
models['Random Forest'] = rf

# Random Forest with hyperparameter tuning
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
}
grid = GridSearchCV(RandomForestClassifier(class_weight='balanced', random_state=42), param_grid, cv=3, scoring='f1', n_jobs=-1)
grid.fit(X_train_res, y_train_res)
best_rf = grid.best_estimator_
results['Random Forest Tuned'] = evaluate_model(best_rf, X_test, y_test)
models['Random Forest Tuned'] = best_rf
print('Best Random Forest Params:', grid.best_params_)

# XGBoost (with scale_pos_weight and advanced hyperparameter tuning)
try:
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    param_dist = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 7, 10],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'scale_pos_weight': [1, scale_pos_weight]
    }
    xgb_search = RandomizedSearchCV(
        xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
        param_distributions=param_dist,
        n_iter=20,
        scoring='f1',
        cv=3,
        n_jobs=-1,
        verbose=1
    )
    xgb_search.fit(X_train_res, y_train_res)
    best_xgb = xgb_search.best_estimator_
    results['XGBoost Tuned'] = evaluate_model(best_xgb, X_test, y_test)
    models['XGBoost Tuned'] = best_xgb
    print('Best XGBoost Params:', xgb_search.best_params_)
    joblib.dump(best_xgb, os.path.join('model_train', 'xgboost_tuned.pkl'))
except Exception as e:
    print(f"XGBoost tuning failed: {e}")

# 4. Select Best Model (by F1, then ROC-AUC)
best_model_name = max(results, key=lambda k: (results[k]['f1'], results[k]['roc_auc'] if results[k]['roc_auc'] is not None else 0))
best_model = models[best_model_name]

print("\n--- Model evaluation results ---")
for name, metrics in results.items():
    print(f"{name}: {metrics}")
print(f"\nBest model: {best_model_name}")

# 5. Save all models and preprocessing objects in model_train folder
os.makedirs('model_train', exist_ok=True)
joblib.dump(lr, os.path.join('model_train', 'logistic_regression.pkl'))
joblib.dump(rf, os.path.join('model_train', 'random_forest.pkl'))
joblib.dump(best_rf, os.path.join('model_train', 'random_forest_tuned.pkl'))
if 'XGBoost' in models:
    joblib.dump(models['XGBoost'], os.path.join('model_train', 'xgboost.pkl'))
joblib.dump(best_model, os.path.join('model_train', 'model.pkl'))
joblib.dump(scaler, os.path.join('model_train', 'scaler.pkl'))
joblib.dump(list(X.columns), os.path.join('model_train', 'model_columns.pkl')) 