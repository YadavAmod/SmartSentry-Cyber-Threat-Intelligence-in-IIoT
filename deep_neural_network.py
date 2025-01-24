import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

def deep_neural_network_model():
    # Load dataset
    data = pd.read_csv('data/live_data_training.csv')
    
    # Check available columns
    available_columns = data.columns.tolist()

    # Ensure 'attack_type' is the correct target column name
    if 'attack_type' not in data.columns:
        return "Error: 'attack_type' column not found in the dataset."

    X = data.drop('attack_type', axis=1)  # Features
    y = data['attack_type']  # Target variable

    # Convert categorical variables in X to numeric using Label Encoding
    label_encoders = {}
    for column in X.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        X[column] = le.fit_transform(X[column])
        label_encoders[column] = le  # Store the encoder for potential inverse transformation later

    # Encode the target variable y
    y_encoder = LabelEncoder()
    y = y_encoder.fit_transform(y)  # Convert attack types to numeric labels

    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Build the neural network
    num_classes = len(set(y))  # Get number of unique classes in y

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax')  # Use softmax for multi-class classification
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    # Train the model
    model.fit(X_train, y_train, epochs=10)

    # Evaluate the model
    loss, accuracy = model.evaluate(X_test, y_test)

    return f"Deep Neural Network Accuracy: {accuracy:.2f}"