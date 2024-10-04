import pickle
import pandas as pd
import numpy as np
from flask import Flask,request,jsonify,render_template


app = Flask(__name__)

ridge_model = pickle.load(open("models/ridge.pkl","rb"))

@app.route("/")
def index():
    return render_template('index.html')  ## it will try to find this page under templates folder


@app.route("/predictdata",methods=['GET','POST'])
def predict_datapoint():
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

        data = [[Temperature,RH,Ws,Rain,FFMC,DMC,ISI,Classes,Region]]
        result=ridge_model.predict(data)

        return render_template('home.html',result=result[0])
    else:
        return render_template("home.html")
        

if __name__ ==  "__main__":
    app.run(host="0.0.0.0",port=8080)
