from flask import Flask, render_template, jsonify
import logging
from datetime import datetime
from utils.data_simulator import IoTDataSimulator
from utils.threat_detector import ThreatDetector
from utils.model_trainer import ModelTrainer

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# Initialize components
simulator = IoTDataSimulator()
detector = ThreatDetector()
trainer = ModelTrainer()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/simulate')
def simulate_data():
    data = simulator.generate_data()
    prediction = detector.detect_threats(data)
    return jsonify({
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'data': data.to_dict('records'),
        'prediction': prediction
    })

if __name__ == '__main__':
    app.run(debug=True)