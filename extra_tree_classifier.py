import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

def extra_tree_classifier_model():
    # Load dataset
    data = pd.read_csv('data/live_data_training.csv')
    
    # Check available columns
    available_columns = data.columns.tolist()

    # Ensure 'attack_type' is the correct target column name
    if 'attack_type' not in data.columns:
        return "Error: 'attack_type' column not found in the dataset."

    X = data.drop('attack_type', axis=1)  # Features
    y = data['attack_type']  # Target variable

    # Convert categorical variables to numeric using Label Encoding
    label_encoders = {}
    for column in X.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        X[column] = le.fit_transform(X[column])
        label_encoders[column] = le  # Store the encoder for potential inverse transformation later

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train Extra Trees Classifier model
    model = ExtraTreesClassifier()
    model.fit(X_train, y_train)

    # Predictions and evaluation
    predictions = model.predict(X_test)
    
    report = classification_report(y_test, predictions)
    conf_matrix = confusion_matrix(y_test, predictions)

    return f"Available columns: {available_columns}\n\nExtra Tree Classifier Classification Report:\n{report}\nConfusion Matrix:\n{conf_matrix}"