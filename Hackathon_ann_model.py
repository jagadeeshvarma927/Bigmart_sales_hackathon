import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import warnings
warnings.filterwarnings('ignore')

# Global variables to store encoders and scaler
label_encoders = {}
scaler = StandardScaler()

def load_data(train_path, test_path):
    """Load training and test datasets"""
    try:
        train_data = pd.read_csv(train_path)
        test_data = pd.read_csv(test_path)
        print(f"Training data shape: {train_data.shape}")
        print(f"Test data shape: {test_data.shape}")
        return train_data, test_data
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None

def explore_data(train_data):
    """Basic data exploration"""
    print("\n=== TRAINING DATA INFO ===")
    print(train_data.info())
    print("\n=== MISSING VALUES ===")
    print(train_data.isnull().sum())
    print("\n=== TARGET VARIABLE STATS ===")
    print(train_data['Item_Outlet_Sales'].describe())

def handle_missing_values(df):
    """Handle missing values in the dataset"""
    df = df.copy()
    
    # Handle Item_Weight missing values
    if 'Item_Weight' in df.columns:
        df['Item_Weight'] = df.groupby('Item_Identifier')['Item_Weight'].transform(
            lambda x: x.fillna(x.median()) if not x.isna().all() else x.fillna(df['Item_Weight'].median())
        )
    
    # Handle Outlet_Size missing values
    if 'Outlet_Size' in df.columns:
        df['Outlet_Size'] = df.groupby('Outlet_Type')['Outlet_Size'].transform(
            lambda x: x.fillna(x.mode()[0]) if not x.mode().empty else x
        )
        df['Outlet_Size'].fillna(df['Outlet_Size'].mode()[0], inplace=True)
    
    return df

def clean_item_fat_content(df):
    """Standardize Item_Fat_Content values"""
    df = df.copy()
    if 'Item_Fat_Content' in df.columns:
        df['Item_Fat_Content'] = df['Item_Fat_Content'].replace({
            'low fat': 'Low Fat',
            'LF': 'Low Fat',
            'reg': 'Regular'
        })
        
        # Non-consumable items should not have fat content
        if 'Item_Type' in df.columns:
            non_consumable_items = ['Health and Hygiene', 'Household', 'Others']
            df.loc[df['Item_Type'].isin(non_consumable_items), 'Item_Fat_Content'] = 'Non-Edible'
    
    return df

def create_new_features(df):
    """Create new features from existing ones"""
    df = df.copy()
    
    # Extract item category from Item_Identifier
    df['Item_Category'] = df['Item_Identifier'].str[:2]
    
    # Years of operation (assuming current year is 2013 as per competition)
    if 'Outlet_Establishment_Year' in df.columns:
        df['Outlet_Years'] = 2013 - df['Outlet_Establishment_Year']
    
    # Item visibility features
    if 'Item_Visibility' in df.columns:
        # Replace 0 visibility with mean visibility for that item
        item_visibility_mean = df.groupby('Item_Identifier')['Item_Visibility'].mean()
        df['Item_Visibility'] = df.apply(
            lambda x: item_visibility_mean[x['Item_Identifier']] if x['Item_Visibility'] == 0 
            else x['Item_Visibility'], axis=1
        )
        
        # Visibility ratio compared to average
        df['Item_Visibility_Ratio'] = df['Item_Visibility'] / df.groupby('Item_Identifier')['Item_Visibility'].transform('mean')
    
    # Price per unit weight
    if 'Item_Weight' in df.columns and 'Item_MRP' in df.columns:
        df['Price_per_Weight'] = df['Item_MRP'] / df['Item_Weight']
    
    # Outlet type and size combination
    if 'Outlet_Type' in df.columns and 'Outlet_Size' in df.columns:
        df['Outlet_Type_Size'] = df['Outlet_Type'] + '_' + df['Outlet_Size'].fillna('Unknown')
    
    return df

def prepare_features(df, is_training=True):
    """Complete feature preparation pipeline"""
    print(f"Preparing features for {'training' if is_training else 'test'} data...")
    
    # Apply all feature engineering steps
    df = handle_missing_values(df)
    df = clean_item_fat_content(df)
    df = create_new_features(df)
    
    print(f"Feature preparation complete. Shape: {df.shape}")
    return df

def encode_categorical_features(train_df, test_df):
    """Encode categorical features using LabelEncoder"""
    global label_encoders
    
    # Identify categorical columns
    categorical_cols = [
        'Item_Fat_Content', 'Item_Type', 'Outlet_Identifier', 
        'Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type',
        'Item_Category', 'Outlet_Type_Size'
    ]
    
    # Ensure all categorical columns exist
    categorical_cols = [col for col in categorical_cols if col in train_df.columns]
    
    train_encoded = train_df.copy()
    test_encoded = test_df.copy()
    
    # Combine train and test for consistent encoding
    for col in categorical_cols:
        # Combine values from both datasets
        combined_values = pd.concat([train_df[col], test_df[col]]).astype(str)
        
        # Create or use existing encoder
        if col not in label_encoders:
            label_encoders[col] = LabelEncoder()
            label_encoders[col].fit(combined_values)
        
        # Encode both datasets
        train_encoded[col + '_encoded'] = label_encoders[col].transform(train_df[col].astype(str))
        test_encoded[col + '_encoded'] = label_encoders[col].transform(test_df[col].astype(str))
    
    print(f"Encoded {len(categorical_cols)} categorical features")
    return train_encoded, test_encoded

def prepare_model_data(train_df, test_df):
    """Prepare data for model training"""
    global scaler
    
    # Define feature columns
    feature_cols = [
        'Item_Weight', 'Item_Visibility', 'Item_MRP', 'Outlet_Establishment_Year',
        'Outlet_Years', 'Item_Visibility_Ratio', 'Price_per_Weight'
    ]
    
    # Add encoded categorical features
    encoded_cols = [col for col in train_df.columns if col.endswith('_encoded')]
    feature_cols.extend(encoded_cols)
    
    # Filter existing columns
    feature_cols = [col for col in feature_cols if col in train_df.columns]
    
    print(f"Selected {len(feature_cols)} features for modeling")
    
    # Prepare training data
    X_train = train_df[feature_cols]
    y_train = train_df['Item_Outlet_Sales']
    
    # Prepare test data
    X_test = test_df[feature_cols]
    
    # Handle any remaining missing values
    X_train = X_train.fillna(X_train.median())
    X_test = X_test.fillna(X_train.median())  # Use train median for test
    
    # Scale features
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, y_train, X_test_scaled, feature_cols

def build_ann_model(input_dim):
    """Build and compile ANN model using Keras"""
    model = keras.Sequential([
        layers.Dense(512, activation='relu', input_dim=input_dim),
        layers.Dropout(0.3),
        layers.BatchNormalization(),
        
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.3),
        layers.BatchNormalization(),
        
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.2),
        layers.BatchNormalization(),
        
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.2),
        
        layers.Dense(32, activation='relu'),
        layers.Dense(1, activation='linear')  # Output layer for regression
    ])
    
    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    
    print("ANN model built successfully")
    print(f"Model has {model.count_params()} parameters")
    return model

def train_model(model, X_train, y_train, validation_split=0.2, epochs=100, batch_size=32):
    """Train the ANN model"""
    print("Starting model training...")
    
    # Callbacks
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=15, restore_best_weights=True
    )
    
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6
    )
    
    # Train model
    history = model.fit(
        X_train, y_train,
        validation_split=validation_split,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )
    
    print("Model training completed")
    return history

def make_predictions(model, X_test):
    """Make predictions using trained model"""
    print("Making predictions...")
    predictions = model.predict(X_test)
    return predictions.flatten()

def create_submission_file(test_df, predictions, filename='submission.csv'):
    """Create submission file in required format"""
    submission = pd.DataFrame({
        'Item_Identifier': test_df['Item_Identifier'],
        'Outlet_Identifier': test_df['Outlet_Identifier'],
        'Item_Outlet_Sales': predictions
    })
    
    submission.to_csv(filename, index=False)
    print(f"Submission file saved as '{filename}'")
    print(f"Submission shape: {submission.shape}")
    print("\nSample predictions:")
    print(submission.head(10))
    
    return submission

def evaluate_model(model, X_train, y_train, validation_split=0.2):
    """Evaluate model performance"""
    # Split data for evaluation
    X_train_eval, X_val_eval, y_train_eval, y_val_eval = train_test_split(
        X_train, y_train, test_size=validation_split, random_state=42
    )
    
    # Make predictions
    train_pred = model.predict(X_train_eval).flatten()
    val_pred = model.predict(X_val_eval).flatten()
    
    # Calculate metrics
    train_rmse = np.sqrt(mean_squared_error(y_train_eval, train_pred))
    val_rmse = np.sqrt(mean_squared_error(y_val_eval, val_pred))
    
    print(f"\nModel Evaluation:")
    print(f"Training RMSE: {train_rmse:.2f}")
    print(f"Validation RMSE: {val_rmse:.2f}")
    
    return train_rmse, val_rmse

# Main execution pipeline
def run_complete_pipeline():
    """Run the complete pipeline from data loading to submission"""
    
    # Step 1: Load data
    print("Step 1: Loading data...")
    train_path = r"C:\Users\canara\Downloads\train_v9rqX0R.csv"
    test_path = r"C:\Users\canara\Downloads\test_AbJTz2l.csv"
    
    train_df, test_df = load_data(train_path, test_path)
    
    if train_df is None or test_df is None:
        print("Failed to load data. Please check file paths.")
        return None, None
    
    # Step 2: Explore data
    print("\nStep 2: Exploring data...")
    explore_data(train_df)
    
    # Step 3: Feature engineering
    print("\nStep 3: Feature engineering...")
    train_processed = prepare_features(train_df, is_training=True)
    test_processed = prepare_features(test_df, is_training=False)
    
    # Step 4: Encode features
    print("\nStep 4: Encoding features...")
    train_encoded, test_encoded = encode_categorical_features(train_processed, test_processed)
    
    # Step 5: Prepare model data
    print("\nStep 5: Preparing model data...")
    X_train, y_train, X_test, feature_cols = prepare_model_data(train_encoded, test_encoded)
    
    print(f"Training features shape: {X_train.shape}")
    print(f"Test features shape: {X_test.shape}")
    
    # Step 6: Build model
    print("\nStep 6: Building ANN model...")
    model = build_ann_model(X_train.shape[1])
    
    # Step 7: Train model
    print("\nStep 7: Training model...")
    history = train_model(model, X_train, y_train, epochs=100)
    
    # Step 8: Evaluate model
    print("\nStep 8: Evaluating model...")
    evaluate_model(model, X_train, y_train)
    
    # Step 9: Make predictions
    print("\nStep 9: Making predictions...")
    predictions = make_predictions(model, X_test)
    
    # Step 10: Create submission
    print("\nStep 10: Creating submission file...")
    submission = create_submission_file(test_df, predictions, r'C:\Users\canara\Downloads\bigmart_submission_ann.csv')
    
    print("\n" + "="*50)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*50)
    
    return model, submission



# Run the complete pipeline
if __name__ == "__main__":
    model, submission = run_complete_pipeline()