from flask import Flask, request, jsonify
import pickle
import numpy as np
from flask_cors import CORS

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

with open('../../model.pkl', 'rb') as file:
    model = pickle.load(file)
@app.route('/')
def hello():
    return 'Hello, World!'

@app.route('/predict', methods=['POST'])
def prediction():
    data = request.get_json()
    input_data = np.array(data['ip'])
    prediction = model.predict(input_data)
    opdata = prediction.tolist()
    return jsonify({'output' : opdata})
    
if __name__ == '__main__':
    app.run(debug=True)