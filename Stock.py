import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data
import yfinance as yf
from datetime import date
from fbprophet import Prophet
from plotly import graph_objs as go
from keras.models import load_model
from fbprophet.plot import plot_plotly

import streamlit as st

start='1980-01-01'
end=date.today().strftime("%Y-%m-%d")

st.title('*STOCKSIM PREDICTIONS*')
user_input=st.text_input('Enter the Stock Ticker','TSLA')
df=data.DataReader(user_input,'yahoo',start,end)
st.subheader('Current data')
st.write(df.tail(5))
st.subheader('Overview')
st.write(df.describe())
st.subheader('Current Intra-day data')
test = yf.download(tickers=user_input, period='5d', interval='5m')
st.write(test.tail(5))
df.reset_index(inplace=True)

st.subheader('Closing Price vs Time Chart')
fig=go.Figure()
fig.add_trace(go.Scatter(x=df['Date'],y=df['Open'],name='stock_open'))
fig.add_trace(go.Scatter(x=df['Date'],y=df['Close'],name='stock_close'))
plt.plot(df.Close)
fig.layout.update(title_text="Closing Price",xaxis_rangeslider_visible=True)
st.plotly_chart(fig)

# Train data
train=pd.DataFrame(df['Close'])
test = yf.download(tickers=user_input, period='5d', interval='5m')
test=test.reset_index()
test=test.drop(['Datetime','Adj Close'],axis=1)
test=pd.DataFrame(test['Close'])

# Min-max Scaler
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler(feature_range=(0,1))
train1=scaler.fit_transform(train)



#Loading model
model=load_model('/Users/vedantpadole/Desktop/Stock_predictions/Stock-Predictions/keras_model.h5')

#Predictions
prev100=train
final_df=prev100.append(test,ignore_index=True)
input_data=scaler.fit_transform(final_df)
x_test=[]
y_test=[]

for i in range(100,input_data.shape[0]):
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i,0])
x_test,y_test=np.array(x_test),np.array(y_test)

y_pred=model.predict(x_test)

scaler=scaler.scale_
scale_f=1/scaler[0]
y_pred=y_pred*scale_f
y_test=y_test*scale_f

st.subheader('Training')
fig2=plt.figure(figsize=(12,6))
plt.plot(y_test,'green',label='Original')
plt.plot(y_pred,'purple',label='Prediction')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)

st.subheader('Prediction')
n_years=st.slider("Years of prediction")
period=n_years*365
df_train=df[['Date','Close']]
df_train=df_train.rename(columns={"Date":"ds","Close":"y"})
m=Prophet()
m.fit(df_train)
future=m.make_future_dataframe(periods=period)
forecast=m.predict(future)
st.write(forecast.tail())

st.write('Final Forcast Data')
fig1=plot_plotly(m,forecast)
st.plotly_chart(fig1)

# Expected Current value
st.subheader('EXPECTED CURRENT VALUE')
x=float(y_pred[-1])
st.text(x)

