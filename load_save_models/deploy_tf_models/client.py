import numpy as np 
from predict_client.prod_client import ProdClient 
from flask import Flask, render_template, url_for
from flask import request 
from flask import jsonify

HOST = '192.168.99.100:9000'
MODEL_NAME = 'test'
MODEL_VERSION = 1

app = Flask(__name__)
client = ProdClient(HOST, MODEL_NAME, MODEL_VERSION)

def convert_data(raw_data):
    return np.array(raw_data, dtype=np.float32)

def get_prediction_from_model(data):
    req_data = [{'in_tensor_name': 'inputs', 'in_tensor_dtype': 'DT_FLOAT', 'data': data}]

    prediction = client.predict(req_data, request_timeout=10)

    return prediction

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/prediction", methods=['POST'])
def get_prediction():
    req_data = request.get_json()
    print("Request_data: ",req_data)
    raw_data = req_data['data']

    data = convert_data(raw_data)
    prediction = get_prediction_from_model(data)

    return jsonify({
        'predictions': prediction['outputs'].tolist()
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000)

