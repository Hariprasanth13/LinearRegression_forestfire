from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

application = Flask(__name__)
app = application

ridge_regressor = pickle.load(open('models/ridge.pkl', 'rb'))
scaler_reg = pickle.load(open('models/scaler.pkl', 'rb'))


@app.route("/")
def hello_world():
    return render_template('index.html')

@app.route("/predictdata",methods = ['GET','POST'])
def predict():
    if request.method == 'POST':
        Temperature=float(request.form.get('Temperature'))
        RH = float(request.form.get('RH'))
        Ws = float(request.form.get('Ws'))
        Rain = float(request.form.get('Rain'))
        FFMC = float(request.form.get('FFMC'))
        DMC = float(request.form.get('DMC'))
        ISI = float(request.form.get('ISI'))
        Classes = float(request.form.get('Classes'))
        Region = float(request.form.get('Region'))

        scaled_data = scaler_reg.transform([[Temperature,RH,Ws,Rain,FFMC,DMC,ISI,Classes,Region]])
        result = ridge_regressor.predict(scaled_data)
        return render_template('home.html',result = result[0])

    else:
        return render_template('home.html')

if __name__=="__main__":
    app.run(host="0.0.0.0")
