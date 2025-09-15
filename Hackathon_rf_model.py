# -----------------------------------------
# RandomForest with Hyperparameter Tuning + KFold CV
# -----------------------------------------
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Load dataset
df = pd.read_csv(r"C:\Users\canara\Downloads\train_v9rqX0R.csv") # Assuming your dataset is named 'train.csv'

# -------------------------------
# 1. Data Cleaning
# -------------------------------
df['Item_Fat_Content'] = df['Item_Fat_Content'].replace({
    'LF': 'Low Fat',
    'low fat': 'Low Fat',
    'reg': 'Regular'
})

df['Item_Weight'].fillna(df['Item_Weight'].mean(), inplace=True)
df['Outlet_Size'].fillna(df['Outlet_Size'].mode()[0], inplace=True)
df['Outlet_Age'] = 2013 - df['Outlet_Establishment_Year']

# -------------------------------
# 2. Feature & Target Split
# -------------------------------
X = df.drop(columns=['Item_Outlet_Sales'])
y = df['Item_Outlet_Sales']

# Exclude 'Item_Identifier' and 'Outlet_Identifier' from the columns to be dropped from X
X_processed = X.drop(columns=['Item_Identifier', 'Outlet_Identifier', 'Outlet_Establishment_Year'], errors='ignore')

# Encode categorical variables
cat_cols = X_processed.select_dtypes(include=['object']).columns
le = LabelEncoder()
for col in cat_cols:
    X_processed[col] = le.fit_transform(X_processed[col])

# For now, X_processed is our feature set.
X = X_processed

# -------------------------------
# 3. Train/Validation Split
# -------------------------------
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# 4. Hyperparameter Tuning with CV
# -------------------------------
param_dist = {
    'n_estimators': [200, 500, 600, 800, 1000],
    'max_depth': [5, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', 0.3, 0.5],
    'bootstrap': [True, False]
}

# 5-fold cross-validation
kf = KFold(n_splits=10, shuffle=True, random_state=42)

rf = RandomForestRegressor(random_state=42, n_jobs=-1)

random_search = RandomizedSearchCV(
    estimator=rf,
    param_distributions=param_dist,
    n_iter=20,              # number of random combinations to try
    cv=kf,
    scoring='neg_mean_absolute_error',
    verbose=2,
    random_state=42,
    n_jobs=-1
)

random_search.fit(X_train, y_train)

# -------------------------------
# 5. Best Model Evaluation
# -------------------------------
print("🔥 Best Parameters found:", random_search.best_params_)
best_model = random_search.best_estimator_

# Evaluate on validation set
y_pred = best_model.predict(X_val)
mae = mean_absolute_error(y_val, y_pred)
print(f"✅ Tuned RandomForest MAE: {mae:.2f}")



import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
import numpy as np

# Load datasets
train = pd.read_csv(r"C:\Users\canara\Downloads\train_v9rqX0R.csv")
test = pd.read_csv(r"C:\Users\canara\Downloads\test_AbJTz2l.csv")  # replace with actual test file name

# Combine train and test for consistent preprocessing (e.g., filling missing values)
# This is a common practice to ensure that both datasets are treated the same way.
# We'll separate them again before model training.
combined_df = pd.concat([train.drop(columns=['Item_Outlet_Sales']), test], ignore_index=True)

# -------------------------------
# 1. Data Cleaning 
# -------------------------------
combined_df['Item_Fat_Content'] = combined_df['Item_Fat_Content'].replace({
    'LF': 'Low Fat',
    'low fat': 'Low Fat',
    'reg': 'Regular'
})

combined_df['Item_Weight'].fillna(combined_df['Item_Weight'].mean(), inplace=True)
combined_df['Outlet_Size'].fillna(combined_df['Outlet_Size'].mode()[0], inplace=True)
combined_df['Outlet_Age'] = 2013 - combined_df['Outlet_Establishment_Year']

# Drop the original 'Outlet_Establishment_Year' as 'Outlet_Age' is created
combined_df.drop(columns=['Outlet_Establishment_Year'], inplace=True)

# -------------------------------
# 2. Feature Engineering & Encoding
# -------------------------------

# Identify categorical columns for encoding, EXCLUDING 'Item_Identifier' and 'Outlet_Identifier'
categorical_cols_to_encode = combined_df.select_dtypes(include=["object"]).columns.tolist()
if 'Item_Identifier' in categorical_cols_to_encode:
    categorical_cols_to_encode.remove('Item_Identifier')
if 'Outlet_Identifier' in categorical_cols_to_encode:
    categorical_cols_to_encode.remove('Outlet_Identifier')

# Apply Label Encoding (fit on combined_df and transform)
le_dict = {}
for col in categorical_cols_to_encode:
    le = LabelEncoder()
    combined_df[col] = le.fit_transform(combined_df[col])
    le_dict[col] = le

# Separate back into processed train and test sets
# The original 'Item_Outlet_Sales' from the 'train' DataFrame needs to be re-attached
train_processed = combined_df.iloc[:len(train)].copy()
test_processed = combined_df.iloc[len(train):].copy()

# Separate target and features for the processed training data
X = train_processed.drop(columns=["Item_Identifier", "Outlet_Identifier"]) # Drop identifiers from features
y = train["Item_Outlet_Sales"]

# Prepare the test set for prediction (drop identifiers)
test_for_prediction = test_processed.drop(columns=["Item_Identifier", "Outlet_Identifier"])

# Use the best parameters you found earlier
best_params = {
    'n_estimators': 200,
    'max_depth': 10,
    'min_samples_split': 5,
    'min_samples_leaf': 4,
    'max_features': 0.5,
    'random_state': 42,
    'n_jobs': -1
}

# Train model
model = RandomForestRegressor(**best_params)
model.fit(X, y)

# Predict on test
preds = model.predict(test_for_prediction)

preds = np.maximum(preds, 0)

# Create submission file
submission = pd.DataFrame({
    "Item_Identifier": test["Item_Identifier"], # Use original identifiers from the test set
    "Outlet_Identifier": test["Outlet_Identifier"],
    "Item_Outlet_Sales": preds
})

submission.to_csv(r"C:\Users\canara\Downloads\submission_rf.csv", index=False)
print("✅ Final submission file created: submission_rf.csv")

print(submission.head())
