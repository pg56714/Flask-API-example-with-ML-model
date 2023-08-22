# -*- coding: UTF-8 -*-
import numpy as np
import model  # 讀取py檔案

from flask import Flask, request, jsonify
from flask_cors import CORS   # 開啟對外存取權

app = Flask(__name__)
# 方法一
CORS(app, resources={
     r"/*": {"origins": "https://website-example-with-ml-model-flask-api.onrender.com"}})
# CORS(app, resources={r"/*": {"origins": "http://127.0.0.1:5500"}})

# 方法二
# # Enable CORS for all routes
# @app.after_request
# def after_request(response):
#     response.headers.add('Access-Control-Allow-Origin',
#                          'http://127.0.0.1:5500')
#     response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
#     response.headers.add('Access-Control-Allow-Methods', 'POST')
#     return response


@app.route('/')
def index():
    return 'hello!!'


@app.route('/test', methods=['POST'])
def getResult():
    input = np.array([[5.5, 2.4, 2.7, 1.]])
    result = model.predict(input)
    return jsonify({'result': str(result)})


@app.route('/predict', methods=['POST'])
def postInput():
    # ok
    # # 取得前端傳過來的數值
    # insertValues = request.get_json()
    # # return jsonify(insertValues)
    # x1 = float(insertValues['sepalLengthCm'])
    # x2 = float(insertValues['sepalWidthCm'])
    # x3 = float(insertValues['petalLengthCm'])
    # x4 = float(insertValues['petalWidthCm'])
    # input = np.array([[x1, x2, x3, x4]])  # 要回傳二維
    # result = model.predict(input)
    # return jsonify({'return': str(result)})

    # ok
    # 確保輸入的資料是XGBoost可以接受的格式
    # Get the data from the frontend
    insertValues = request.get_json()

    # Extract input features and convert to float
    try:
        x1 = float(insertValues['sepalLengthCm'])
        x2 = float(insertValues['sepalWidthCm'])
        x3 = float(insertValues['petalLengthCm'])
        x4 = float(insertValues['petalWidthCm'])
    except ValueError:
        return jsonify({'error': 'Invalid input data'})

    input_data = np.array([[x1, x2, x3, x4]])

    # Check for missing values and handle as needed
    if np.isnan(input_data).any():
        return jsonify({'error': 'Missing values in input data'})

    # Make predictions with the XGBoost model
    try:
        result = model.predict(input_data)
    except Exception as e:
        return jsonify({'error': str(e)})

    return jsonify({'result': result.tolist()})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000, debug=True)
