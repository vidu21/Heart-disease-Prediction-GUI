
import numpy as np
from flask import Flask, request, jsonify, render_template, url_for
import pickle

app = Flask(__name__)
model = pickle.load(open('data.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    to_predict = request.form.to_dict()
    to_predict = list(to_predict.values())
    to_predict = list(map(int, to_predict))
    final_features = np.array(to_predict).reshape(1, 10)
    prediction = model.predict(final_features)

    output = round(prediction[0])
    if int(output)==0:
        pred='Your chance to suffer from heart disease is 0. You are not suffering from heart disease'
    else:
        pred='Your chance to suffer from heart disease is 1. You are suffering from heart disease. Consult to the doctor'

    return render_template('result.html', prediction_text= pred)

@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)