import pandas as pd
import numpy as np
import pickle
from flask import Flask,app,jsonify,request, render_template,url_for
import statsmodels.api as sm
import streamlit as st


app=Flask(__name__)

ml_model=pickle.load(open("regmodel.pkl","rb"))
scaling_model=pickle.load(open("scalermodel.pkl","rb"))

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/predict_bike_sharing",methods=['POST'])
def predict_bike_sharing_api():
    data=request.json['data']
    # print(data)
    test_df=pd.DataFrame(data, index=['i',])
    # print(test_df)

    test_df.weathersit.replace({1:'clear',2:'cloudy',3:'light_rain',4:'heavy_rain'},inplace = True)
    test_df.season.replace({1:"spring", 2:"summer", 3:"fall", 4:"winter"},inplace = True)
    test_df.weekday.replace({0: 'sun',1: 'mon',2: 'tue',3: 'wed',4: 'thu',5: 'fri',6: 'sat'},inplace=True)
    test_df.mnth.replace({1: 'jan',2: 'feb',3: 'mar',4: 'apr',5: 'may',6: 'jun',
                    7: 'jul',8: 'aug',9: 'sept',10: 'oct',11: 'nov',12: 'dec'},inplace=True)

    test_df = pd.get_dummies(data=test_df,columns=["season","mnth","weekday"])
    test_df = pd.get_dummies(data=test_df,columns=["weathersit"])

    df=pd.DataFrame(columns=['instant', 'dteday', 'yr', 'holiday', 'workingday', 'temp', 'atemp',
       'hum', 'windspeed', 'casual', 'registered', 'cnt', 'season_fall',
       'season_spring', 'season_summer', 'season_winter', 'mnth_apr',
       'mnth_aug', 'mnth_dec', 'mnth_feb', 'mnth_jan', 'mnth_jul', 'mnth_jun',
       'mnth_mar', 'mnth_may', 'mnth_nov', 'mnth_oct', 'mnth_sept',
       'weekday_fri', 'weekday_mon', 'weekday_sat', 'weekday_sun',
       'weekday_thu', 'weekday_tue', 'weekday_wed', 'weathersit_clear',
       'weathersit_cloudy', 'weathersit_light_rain'])
    # df
    test_df1=pd.concat([df,test_df],axis=0)
    test_df1.fillna(0,inplace=True)
    test_df1.reset_index(inplace=True,drop=True)
    print(test_df1)


    final_columns=['yr', 'holiday', 'windspeed','temp','season_spring',
       'season_summer', 'season_winter', 'mnth_sept',
       'weathersit_cloudy', 'weathersit_light_rain']

    # test_df1=test_df.loc[2:2]
    # test_df1

    #scaling the numerical/continous variables
    numerical_columns=["temp","atemp","hum","windspeed"]
    test_df1[numerical_columns]=scaling_model.transform(test_df1[numerical_columns])

    # Y_test=test_df1[["cnt"]]
    # test_df1.drop(["cnt"],axis=1,inplace=True)
    X_test=test_df1.copy()
    X_test_sm10=sm.add_constant(X_test[final_columns],has_constant='add')
    pred=ml_model.predict(X_test_sm10)
    print(pred.item())
    return jsonify(pred.item())
    # reponse= {"response":1}
    # return reponse


@app.route("/predict_bike_sharing_web",methods=['POST'])
def predict_bike_sharing_api_web():
    # data=request.json['data']
    print(request.form.values())
    data_from_request=[x for x in request.form.values()]
    print(data_from_request)
    data1=[int(x) for x in data_from_request[0:1]]
    data2=[str(x) for x in data_from_request[1:2]]
    data3=[int(x) for x in data_from_request[2:9]]
    data4=[float(x) for x in data_from_request[9:13]]
    data5=[int(x) for x in data_from_request[13:]]


    data=data1+data2+data3+data4+data5

    print(data)
    # response=data
    # # print(data)
    # test_df=pd.DataFrame(data, index=['i',])
    column_data_df=['instant', 'dteday', 'season', 'yr', 'mnth', 'holiday', 'weekday',
       'workingday', 'weathersit', 'temp', 'atemp', 'hum', 'windspeed',
       'casual', 'registered']
    # data=[1.0, '03-01-2018', 1.0, 0.0, 1.0, 0.0, 3.0, 1.0, 1.0, 8.050924, 9.47025, 43.72, 16.636703, 120.0, 1229.0]
    data=np.array(data).reshape(1, -1)
    test_df=pd.DataFrame(data,columns=column_data_df)
    int_columns=['instant', 'season', 'yr', 'mnth', 'holiday', 'weekday',
       'workingday', 'weathersit','casual', 'registered']
    test_df[int_columns]=test_df[int_columns].astype('int')
    
    # print(test_df)

    test_df.weathersit.replace({1:'clear',2:'cloudy',3:'light_rain',4:'heavy_rain'},inplace = True)
    test_df.season.replace({1:"spring", 2:"summer", 3:"fall", 4:"winter"},inplace = True)
    test_df.weekday.replace({0: 'sun',1: 'mon',2: 'tue',3: 'wed',4: 'thu',5: 'fri',6: 'sat'},inplace=True)
    test_df.mnth.replace({1: 'jan',2: 'feb',3: 'mar',4: 'apr',5: 'may',6: 'jun',
                    7: 'jul',8: 'aug',9: 'sept',10: 'oct',11: 'nov',12: 'dec'},inplace=True)

    test_df = pd.get_dummies(data=test_df,columns=["season","mnth","weekday"])
    test_df = pd.get_dummies(data=test_df,columns=["weathersit"])

    df=pd.DataFrame(columns=['instant', 'dteday', 'yr', 'holiday', 'workingday', 'temp', 'atemp',
       'hum', 'windspeed', 'casual', 'registered', 'cnt', 'season_fall',
       'season_spring', 'season_summer', 'season_winter', 'mnth_apr',
       'mnth_aug', 'mnth_dec', 'mnth_feb', 'mnth_jan', 'mnth_jul', 'mnth_jun',
       'mnth_mar', 'mnth_may', 'mnth_nov', 'mnth_oct', 'mnth_sept',
       'weekday_fri', 'weekday_mon', 'weekday_sat', 'weekday_sun',
       'weekday_thu', 'weekday_tue', 'weekday_wed', 'weathersit_clear',
       'weathersit_cloudy', 'weathersit_light_rain'])
    # df
    
    test_df1=pd.concat([df,test_df],axis=0)
    test_df1.fillna(0,inplace=True)
    test_df1.reset_index(inplace=True,drop=True)
    print(test_df1)


    final_columns=['yr', 'holiday', 'windspeed','temp','season_spring',
       'season_summer', 'season_winter', 'mnth_sept',
       'weathersit_cloudy', 'weathersit_light_rain']

    # test_df1=test_df.loc[2:2]
    # test_df1

    #scaling the numerical/continous variables
    numerical_columns=["temp","atemp","hum","windspeed"]
    test_df1[numerical_columns]=scaling_model.transform(test_df1[numerical_columns])

    # Y_test=test_df1[["cnt"]]
    # test_df1.drop(["cnt"],axis=1,inplace=True)
    X_test=test_df1.copy()
    X_test_sm10=sm.add_constant(X_test[final_columns],has_constant='add')
    pred=ml_model.predict(X_test_sm10)
    print(pred.item())
    # return jsonify(pred.item())
    # # response= {"response":1}
    # return response
    return render_template("home.html",prediction_text=f"Bike Sharing Prediction is {pred.item()}")


if __name__=="__main__":
    app.run(debug=True)
