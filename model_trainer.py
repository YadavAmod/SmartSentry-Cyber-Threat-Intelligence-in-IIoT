# utils/model_trainer.py
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import joblib

class ModelTrainer:
    def __init__(self):
        self.models = {
            'random_forest': RandomForestClassifier(),
            'decision_tree': DecisionTreeClassifier(),
            'svm': SVC(),
            'knn': KNeighborsClassifier()
        }
        
    def train_model(self, X_train, y_train, model_name='random_forest'):
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not supported")
            
        model = self.models[model_name]
        model.fit(X_train, y_train)
        
        # Save the trained model
        joblib.dump(model, f'models/{model_name}_model.pkl')
        return model