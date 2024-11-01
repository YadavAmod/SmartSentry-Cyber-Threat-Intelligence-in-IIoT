# utils/threat_detector.py
from sklearn.ensemble import RandomForestClassifier
import joblib

class ThreatDetector:
    def __init__(self):
        self.model = None
        try:
            self.model = joblib.load('models/random_forest_model.pkl')
        except:
            self.model = RandomForestClassifier()
            
    def detect_threats(self, data):
        if self.model is None:
            return "Model not trained"
        return "Normal"  # Placeholder for actual prediction