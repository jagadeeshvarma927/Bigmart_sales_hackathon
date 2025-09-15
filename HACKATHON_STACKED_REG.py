import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def load_dataset(train_path, test_path):
    """
    Load training and test datasets
    """
    print("Loading datasets...")
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    
    print(f"Train shape: {train.shape}")
    print(f"Test shape: {test.shape}")
    print(f"Train columns: {list(train.columns)}")
    
    return train, test

def prepare_features(train, test):
    """
    Feature engineering and preparation with one-hot encoding
    """
    print("Preparing features...")
    
    # Combine train and test for consistent preprocessing
    train['source'] = 'train'
    test['source'] = 'test'
    
    # Add target column to test with dummy values for concatenation
    test['Item_Outlet_Sales'] = 0
    
    combined = pd.concat([train, test], ignore_index=True)
    
    # Handle missing values
    print("Handling missing values...")
    combined['Item_Weight'].fillna(combined['Item_Weight'].median(), inplace=True)
    combined['Outlet_Size'].fillna('Medium', inplace=True)
    
    # Fix inconsistent categories in Item_Fat_Content
    combined['Item_Fat_Content'] = combined['Item_Fat_Content'].replace({
        'low fat': 'Low Fat',
        'LF': 'Low Fat', 
        'reg': 'Regular'
    })
    
    # Create new features
    print("Creating new features...")
    combined['Item_Type_Combined'] = combined['Item_Identifier'].apply(lambda x: x[0:2])
    
    # Set Non-Edible items fat content to Non-Edible
    combined.loc[combined['Item_Type_Combined'].isin(['NC']), 'Item_Fat_Content'] = 'Non-Edible'
    
    # Years of operation
    combined['Years_Operation'] = 2013 - combined['Outlet_Establishment_Year']
    
    # Item visibility - replace 0 with mean visibility of that item type
    print("Processing item visibility...")
    visibility_avg = combined.pivot_table(values='Item_Visibility', index='Item_Identifier', aggfunc='mean')
    missing_visibility = combined['Item_Visibility'] == 0
    for idx in combined[missing_visibility].index:
        item_id = combined.loc[idx, 'Item_Identifier']
        if item_id in visibility_avg.index:
            combined.loc[idx, 'Item_Visibility'] = visibility_avg.loc[item_id, 'Item_Visibility']
    
    # New visibility feature
    item_visibility_mean = combined.groupby(['Item_Identifier'])['Item_Visibility'].mean()
    combined['Item_Visibility_MeanRatio'] = combined.apply(
        lambda row: row['Item_Visibility'] / item_visibility_mean[row['Item_Identifier']], axis=1
    )
    
    # MRP bins
    combined['Item_MRP_Bins'] = pd.cut(combined['Item_MRP'], 
                                      bins=[0, 69, 136, 203, 270], 
                                      labels=['Low', 'Medium', 'High', 'Very_High'])
    
    # One-hot encoding for categorical variables
    print("Applying one-hot encoding...")
    categorical_features = ['Item_Fat_Content', 'Item_Type', 'Outlet_Size', 
                           'Outlet_Location_Type', 'Outlet_Type', 'Item_Type_Combined', 
                           'Item_MRP_Bins']
    
    # Apply one-hot encoding
    combined_encoded = pd.get_dummies(combined, columns=categorical_features, drop_first=False)
    
    # Drop unnecessary columns
    columns_to_drop = ['Item_Identifier', 'Outlet_Identifier', 'Outlet_Establishment_Year']
    combined_encoded = combined_encoded.drop(columns_to_drop, axis=1)
    
    # Split back to train and test
    train_processed = combined_encoded[combined_encoded['source'] == 'train'].copy()
    test_processed = combined_encoded[combined_encoded['source'] == 'test'].copy()
    
    # Remove source column
    train_processed = train_processed.drop('source', axis=1)
    test_processed = test_processed.drop('source', axis=1)
    
    # Remove target from test
    test_processed = test_processed.drop('Item_Outlet_Sales', axis=1)
    
    print(f"Processed train shape: {train_processed.shape}")
    print(f"Processed test shape: {test_processed.shape}")
    
    return train_processed, test_processed

from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LassoCV, RidgeCV
from sklearn.ensemble import StackingRegressor
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import GradientBoostingRegressor

def build_stacked_model(meta_model_type='lasso'):
    """
    Build stacked model with XGBoost, LightGBM, GradientBoosting and meta-regressor
    """
    # Define base estimators
    estimators = [
        ('xgb', xgb.XGBRegressor(
            n_estimators=1000,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        )),
        ('lgb', lgb.LGBMRegressor(
            n_estimators=1000,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )),
        ('gbr', GradientBoostingRegressor(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            random_state=42
        ))
    ]
    
    # Use a more flexible meta-regressor
    if meta_model_type == 'lasso':
        meta_regressor = LassoCV(
            eps=1e-5, 
            n_alphas=50, 
            random_state=42,
            cv=5
        )
    elif meta_model_type == 'ridge':
        meta_regressor = RidgeCV(
            cv=5
        )
    else:
        raise ValueError("meta_model_type must be 'lasso' or 'ridge'")
    
    # Create the StackingRegressor with 10-fold cross-validation
    stacked_model = StackingRegressor(
        estimators=estimators,
        final_estimator=meta_regressor,
        cv=10,  # Increased folds for more robust cross-validation
        n_jobs=-1
    )
    
    return stacked_model

def train_stacked_model(X_train, y_train, meta_model_type='lasso'):
    """meta_regressor
    Train the stacked model
    """
    print("Building and training stacked model...")
    print(f"Using {meta_model_type} as meta-regressor")
    
    # Build stacked model
    stacked_model = build_stacked_model(meta_model_type)
    
    # Train the model
    stacked_model.fit(X_train, y_train)
    
    print("Stacked model training completed!")
    
    return stacked_model

def predict_with_stacked_model(stacked_model, X_test):
    """
    Generate predictions using trained stacked model
    """
    print("Generating predictions with stacked model...")
    predictions = stacked_model.predict(X_test)
    
    return predictions


def evaluate_model(model, X_train, y_train, cv_folds=10):
    """
    Evaluate model using cross-validation with Mean Absolute Error
    """
    print("Evaluating model with cross-validation...")
    
    from sklearn.model_selection import cross_val_score
    
    # Use sklearn's cross_val_score for easier evaluation
    cv_scores = cross_val_score(
        model, X_train, y_train, 
        cv=cv_folds, 
        scoring='neg_mean_absolute_error',
        n_jobs=-1
    )
    
    # Convert negative MAE back to positive
    cv_scores = -cv_scores
    
    for fold, score in enumerate(cv_scores):
        print(f"Fold {fold + 1} MAE: {score:.4f}")
    
    mean_cv_score = np.mean(cv_scores)
    std_cv_score = np.std(cv_scores)
    
    print(f"Mean CV MAE: {mean_cv_score:.4f} (+/- {std_cv_score:.4f})")
    
    return mean_cv_score, std_cv_score


def create_submission_file(test_df, predictions, filename='submission.csv'):
    """
    Create submission file in required format
    """
    print("Creating submission file...")

    predictions = np.maximum(predictions, 0)

    # Create submission dataframe
    submission = pd.DataFrame({
        'Item_Identifier': test_df['Item_Identifier'],
        'Outlet_Identifier': test_df['Outlet_Identifier'],
        'Item_Outlet_Sales': predictions
    })
    
    print(f"Submission shape: {submission.shape}")
    print("Sample submission:")
    print(submission.head())
    
    # Save to CSV
    submission.to_csv(filename, index=False)
    print(f"Submission file saved as: {filename}")
    
    return submission

def main():
    """
    Main function to execute the complete pipeline
    """
    print("="*60)
    print("BIG MART SALES PREDICTION - HACKATHON SOLUTION")
    print("="*60)
    
    # File paths
    train_path = r"C:\Users\canara\Downloads\train_v9rqX0R.csv"
    test_path = r"C:\Users\canara\Downloads\test_AbJTz2l.csv"
    
    # Step 1: Load datasets
    print("\n" + "="*40)
    print("STEP 1: LOADING DATASETS")
    print("="*40)
    train_original, test_original = load_dataset(train_path, test_path)
    
    # Step 2: Feature preparation
    print("\n" + "="*40)
    print("STEP 2: FEATURE ENGINEERING")
    print("="*40)
    train_processed, test_processed = prepare_features(train_original, test_original)
    
    # Prepare X and y
    X_train = train_processed.drop('Item_Outlet_Sales', axis=1)
    y_train = train_processed['Item_Outlet_Sales']
    X_test = test_processed
    
    print(f"\nFinal feature shape:")
    print(f"X_train: {X_train.shape}")
    print(f"y_train: {y_train.shape}")
    print(f"X_test: {X_test.shape}")
    
    # Step 3: Train stacked model
    print("\n" + "="*40)
    print("STEP 3: TRAINING STACKED MODEL")
    print("="*40)
    
    # Try both Lasso and Ridge meta-regressors
    models = {}
    scores = {}
    
    for meta_type in ['lasso', 'ridge']:
        print(f"\n--- Training with {meta_type.upper()} meta-regressor ---")
        model = train_stacked_model(X_train, y_train, meta_model_type=meta_type)
        mean_score, std_score = evaluate_model(model, X_train, y_train, cv_folds=10)
        
        models[meta_type] = model
        scores[meta_type] = mean_score
        
        print(f"{meta_type.upper()} model CV MAE: {mean_score:.4f} (+/- {std_score:.4f})")
    
    # Select best model
    best_meta_type = min(scores, key=scores.get)
    best_model = models[best_meta_type]
    best_score = scores[best_meta_type]
    
    print(f"\n*** BEST MODEL: {best_meta_type.upper()} with CV MAE: {best_score:.4f} ***")
    
    # Step 4: Generate predictions
    print("\n" + "="*40)
    print("STEP 4: GENERATING PREDICTIONS")
    print("="*40)
    
    # Retrain best model on full training data
    print("Retraining best model on full training data...")
    best_model.fit(X_train, y_train)
    
    # Generate test predictions
    test_predictions = predict_with_stacked_model(best_model, X_test)
    
    print(f"Test predictions generated: {len(test_predictions)} samples")
    print(f"Prediction stats:")
    print(f"  Min: {test_predictions.min():.2f}")
    print(f"  Max: {test_predictions.max():.2f}")
    print(f"  Mean: {test_predictions.mean():.2f}")
    print(f"  Std: {test_predictions.std():.2f}")
    
    # Step 5: Create submission file
    print("\n" + "="*40)
    print("STEP 5: CREATING SUBMISSION FILE")
    print("="*40)
    
    submission = create_submission_file(test_original, test_predictions, r'C:\Users\canara\Downloads\bigmart_submission_stacked.csv')
    
    print("\n" + "="*60)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*60)
    print(f"Best Model: {best_meta_type.upper()} meta-regressor")
    print(f"Cross-validation MAE: {best_score:.4f}")
    print(f"Submission file: bigmart_submission.csv")
    print(f"Submission shape: {submission.shape}")
    
    return best_model, submission

# Run the complete pipeline
if __name__ == "__main__":
    model, submission = main()