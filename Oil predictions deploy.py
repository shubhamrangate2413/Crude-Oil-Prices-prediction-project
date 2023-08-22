"""
Created by 27-04-2023
@author: Akash Shinde
"""
import keras.models
import numpy as np
import pandas as pd
import streamlit as st
import joblib
import pickle
from sklearn.preprocessing import MinMaxScaler
from keras.layers import Dense
from keras.layers import Input, LSTM
from keras.models import Model
import h5py
from datetime import timedelta, date

SC = joblib.load("SC")
model = keras.models.load_model("Model.keras")


# It will predict from 26 July to 24 August 2023.


def Prediction(x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18, x19, x20, x21, x22, x23, x24, x25, x26, x27, x28, x29, x30):
    A = [x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18, x19, x20, x21, x22, x23, x24, x25, x26, x27, x28, x29, x30]
    A = np.array(A)
    # A = A.reshape(1, 30, 1)
    pred = model.predict(A)
    pred = SC.inverse_transform(pred.reshape(64, 30))
    prediction = list(pred[1])
    return prediction


def main():
    st.title("Oil Stock Price Prediction Gr-6")
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style ="color:white;text-align:center;>Streamlit Bank Authenticator ML App </h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)
    x1 = st.number_input('Enter value:', value=69.370000)
    x2 = st.number_input('Enter value:', value=68.444500)
    x3 = st.number_input('Enter value:', value=69.560000)
    x4 = st.number_input('Enter value:', value=69.860000)
    x5 = st.number_input('Enter value:', value=70.640000)
    x6 = st.number_input('Enter value:', value=70.356667)
    x7 = st.number_input('Enter value:', value=70.073333)
    x8 = st.number_input('Enter value:', value=69.790000)
    x9 = st.number_input('Enter value:', value=71.060000)
    x10 = st.number_input('Enter value:', value=71.790000)
    x11 = st.number_input('Enter value:', value=71.800000)
    x12 = st.number_input('Enter value:', value=73.860000)
    x13 = st.number_input('Enter value:', value=73.570000)
    x14 = st.number_input('Enter value:', value=73.280000)
    x15 = st.number_input('Enter value:', value=72.990000)
    x16 = st.number_input('Enter value:', value=74.830000)
    x17 = st.number_input('Enter value:', value=75.750000)
    x18 = st.number_input('Enter value:', value=76.890000)
    x19 = st.number_input('Enter value:', value=75.420000)
    x20 = st.number_input('Enter value:', value=74.996667)
    x21 = st.number_input('Enter value:', value=74.573333)
    x22 = st.number_input('Enter value:', value=74.150000)
    x23 = st.number_input('Enter value:', value=75.750000)
    x24 = st.number_input('Enter value:', value=75.350000)
    x25 = st.number_input('Enter value:', value=75.630000)
    x26 = st.number_input('Enter value:', value=77.070000)
    x27 = st.number_input('Enter value:', value=77.626667)
    x28 = st.number_input('Enter value:', value=78.183333)
    x29 = st.number_input('Enter value:', value=78.740000)
    x30 = st.number_input('Enter value:', value=79.640000)
    result1 = ""
    if st.button("Predict"):
        result1 = Prediction(x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18, x19, x20, x21, x22, x23, x24, x25, x26, x27, x28, x29, x30)
    st.success('The predicted values is {}'.format(result1))


if __name__ == '__main__':
    main()
