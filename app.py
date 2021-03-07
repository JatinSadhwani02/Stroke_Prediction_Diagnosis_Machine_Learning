  
from flask import Flask, render_template, request
import joblib
import os
import  numpy as np
import pickle

app= Flask(__name__)

@app.route("/")
def index():
    return render_template("home.html")

@app.route("/result",methods=['POST','GET'])
def result():
    gender=(request.form['gender'])
    age=int(request.form['age'])
    hypertension=int(request.form['hypertension'])
    heart_disease = int(request.form['heart_disease'])
    ever_married = (request.form['ever_married'])
    work_type = (request.form['work_type'])
    Residence_type = (request.form['Residence_type'])
    avg_glucose_level = float(request.form['avg_glucose_level'])
    bmi = float(request.form['bmi'])
    smoking_status = (request.form['smoking_status'])

    if gender=='m' or gender=='M' or gender=='Male' or gender=='male':
        gender=1
    else:
        gender=0

    if work_type=='p' or work_type=='P':
        work_type=2
    elif work_type=='se' or work_type=='SE':
        work_type=3
    elif work_type == 'GJ' or work_type=='gj':
        work_type=0
    elif work_type== 'c' or work_type=='C':
        work_type=4
    else:
        work_type=1

    if Residence_type=='U' or Residence_type=='u':
        Residence_type = 1
    elif Residence_type=='R' or Residence_type=='r' :
        Residence_type=0

    if smoking_status=='fs' or smoking_status =='FS' :
        smoking_status=1
    elif smoking_status=='ns' or smoking_status =='NS' :
        smoking_status=2
    elif smoking_status=='s' or smoking_status =='S' :
        smoking_status=3
    elif smoking_status=='u' or smoking_status =='U' :
        smoking_status=0

    if ever_married=='y' or ever_married=='Y':
        ever_married=1
    elif ever_married=='n' or ever_married=='N':
        ever_married=0

    x=np.array([gender,age,hypertension,heart_disease,ever_married,work_type,Residence_type,
                avg_glucose_level,bmi,smoking_status]).reshape(1,-1)

    scaler_path=os.path.join('C:/Users/om sai ram/Desktop/DS/stroke prediction/','model/scaler.pkl')
    scaler=None
    with open(scaler_path,'rb') as scaler_file:
        scaler=pickle.load(scaler_file)

    x=scaler.transform(x)

    # model_path=os.path.join('C:/Users/om sai ram/Desktop/DS/stroke prediction/','models/decision_tree_model.sav')
    model_path = 'C:/Users/om sai ram/Desktop/DS/stroke prediction/model/decision_tree_model.sav'
    dt=joblib.load(model_path)

    Y_pred=dt.predict(x)

    # for No Stroke Risk
    if Y_pred==0:
        return render_template('nostroke.html')
    else:
        return render_template('stroke.html')

if __name__=="__main__":
    app.run(debug=True,port=7384)