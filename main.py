import streamlit as st
from datetime import date

import yfinance as yf
from fbprophet import Prophet
from fbprophet.plot import plot_plotly
from plotly import graph_objs as go

START = "2015-01-01"
TODAY = date.today()

st.title("Stock Prediction App")

stocks = ("AAPL", "GOOG", "MSFT", "GME", "L.TO", "AC.TO")
selected_stocks = st.selectbox("Select dataset for prediction", stocks)

n_years = st.slider("Years of prediction", 1, 4)
period = n_years * 365

# This caches the data. It is important so we do not use resources again that was already used. If Google stocks get called, it caches it and we do not need to reload it.


@st.cache
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data


data_load_state = st.text("Load data...")
data = load_data(selected_stocks)
data_load_state.text("Loading data...done!")

# Raw data of given selected_stocks
st.subheader('Raw data')
st.write(data.tail())

# Plot a nice graph using the API


def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'],
                             y=data['Open'], name='Stock Open'))
    fig.add_trace(go.Scatter(x=data['Date'],
                             y=data['Close'], name='Stock Closed'))
    fig.layout.update(title_text="Time Series Data",
                      xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)


plot_raw_data()


# Forecasting
df_train = data[['Date', 'Close']]
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

m = Prophet()
m.fit(df_train)
future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

# Raw data of given selected_stocks
st.subheader('Forecast data')
st.write(forecast.tail())

st.write('forecast data')
fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1)

st.write('forecast components')
fig2 = m.plot_components(forecast)
st.write(fig2)
