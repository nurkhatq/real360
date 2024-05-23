# model.py
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import joblib

def train_and_save_model():
    # Sample dataset
    

    # Create DataFrame
    df = pd.read_csv('cleaned_krysha.csv')
    df = df.drop(columns=['Complex Name'])
    # Convert 'Area' column to numeric and remove 'м²'
    df['Area'] = df['Area'].str.replace(' м²', '').astype(float)

    # Extract the number of rooms from 'Property Type'
    df['Rooms'] = df['Property Type'].str.extract(r'(\d+)').astype(int)

    # Drop unnecessary columns
    df.drop(columns=['Property Type'], inplace=True)

    # One-hot encode categorical variables
    df = pd.get_dummies(df, columns=['Region', 'Home Type'], drop_first=True)

    # Separate features (X) and target variable (y)
    X = df.drop(columns=['Price'])
    y = df['Price']

    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Initialize the RandomForestRegressor model
    model = RandomForestRegressor(n_estimators=100, random_state=42)

    # Train the model
    model.fit(X_scaled, y)

    # Save the model and scaler
    joblib.dump(model, 'model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    joblib.dump(X.columns, 'columns.pkl')

def predict_price(new_data):
    model = joblib.load('model.pkl')
    scaler = joblib.load('scaler.pkl')
    columns = joblib.load('columns.pkl')

    new_df = pd.DataFrame(new_data)

    # Preprocess new data similarly
    new_df['Rooms'] = new_df['Property Type']
    new_df.drop(columns=['Property Type'], inplace=True)
    new_df = pd.get_dummies(new_df, columns=['Region', 'Home Type'], drop_first=False)

    # Ensure the new data has the same columns as the training data
    for col in columns:
        if col not in new_df.columns:
            new_df[col] = 0

    # Scale the new data
    new_df = new_df[columns]

    new_df_scaled = scaler.transform(new_df)

    # Predict the price for new data
    new_pred = model.predict(new_df_scaled)
    return new_pred[0]

# Uncomment the line below to train and save the model when you first run the script
train_and_save_model()
