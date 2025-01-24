import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

def decision_tree_model():
    # Load dataset with error handling
    try:
        data = pd.read_csv('data/live_data_training.csv')
    except FileNotFoundError:
        return "Error: The specified file was not found."

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

    # Initialize and train Decision Tree model
    model = DecisionTreeClassifier(random_state=42)
    
    try:
        model.fit(X_train, y_train)
    except Exception as e:
        return f"Error during model training: {e}"

    # Predictions and evaluation
    predictions = model.predict(X_test)
    
    report = classification_report(y_test, predictions)
    conf_matrix = confusion_matrix(y_test, predictions)

    return f"Available columns: {available_columns}\n\nDecision Tree Classification Report:\n{report}\nConfusion Matrix:\n{conf_matrix}"