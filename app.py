from flask import Flask, render_template, jsonify
# Importing  model functions here
from models import random_forest as rf
from models import decision_tree as dt
from models import extra_tree_classifier as etc
from models import svm
from models import knn
from models import deep_neural_network as dnn

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/run_model/<model_name>', methods=['GET'])
def run_model(model_name):
    if model_name == 'random_forest':
        result = rf.random_forest_model()  # Ensuring this function returns a string or dict with results
    elif model_name == 'decision_tree':
        result = dt.decision_tree_model()
    elif model_name == 'extra_tree_classifier':
        result = etc.extra_tree_classifier_model()
    elif model_name == 'svm':
        result = svm.svm_model()
    elif model_name == 'knn':
        result = knn.knn_model()
    elif model_name == 'deep_neural_network':
        result = dnn.deep_neural_network_model()
    else:
        result = "Invalid model name!"

    return jsonify(result=result)

if __name__ == "__main__":
    app.run(debug=True)