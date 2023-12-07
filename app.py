import streamlit as st
import pickle
import numpy as np
import pandas as pd

df = pd.read_csv("df.csv")
pipe = pickle.load(open("Prediction_model.pkl", "rb"))
st.title("Laptop Price Predictor")

Company_Name = st.selectbox('Company', df['Company'].unique())

Laptop_Type = st.selectbox("Laptop Type", df['TypeName'].unique())

RAM = st.selectbox("Ram(in GB)", [2, 4, 6, 8, 12, 16, 24, 32, 64])

Weight = st.number_input("Weight(in kg)")


Touchscreen = st.selectbox("TouchScreen", ['No', 'Yes'])


IPS = st.selectbox("IPS", ['No', 'Yes'])


Size_of_screen = st.number_input('Screen Size')


Resolution = st.selectbox('Screen Resolution', [
                          '1920x1080', '1366x768', '1600x900', '3200x1800', '2880x1800', '2560x1600', '2304x1440'])

CPU = st.selectbox('CPU', df['Cpu brand'].unique())

Harddisk = st.selectbox('HDD(in GB)', [0,128, 256, 512, 1024, 2048])

SSD = st.selectbox('SSD(in GB)', [ 128, 256, 512, 1024])

GPU = st.selectbox('Gpraphic card', df['Gpu brand'].unique())

OperatingSys = st.selectbox('Operating system', df['os'].unique())


if st.button('Predict'):
    ppi = None
    if Touchscreen == "Yes":
        Touchscreen = 1
    else:
        Touchscreen = 0
    if IPS == "Yes":
        IPS = 1
    else:
        IPS = 0
        
        
    X_res = int(Resolution.split('x')[0])
    Y_res = int(Resolution.split('x')[1])
    
    ppi = ((X_res ** 2) + (Y_res**2)) ** 0.5 / Size_of_screen
    query = np.array([Company_Name, Laptop_Type, RAM, Weight,
                     Touchscreen, IPS, ppi, CPU, Harddisk, SSD, GPU, OperatingSys])
    query = query.reshape(1, 12)
    prediction = str(int(np.exp(pipe.predict(query)[0])))
    st.title("Predicted Price of Laptop is $ " + prediction)
