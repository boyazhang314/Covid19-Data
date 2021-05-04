import streamlit as st
import pandas as pd
import numpy as np

st.title('Covid-19 Data: Canada')

# data sets
covid_data = "WHO-COVID-19-global-data.csv"
vacine_data = "vaccination-data.csv"

# load data
@st.cache
def loadDataC(data):
    dt = pd.read_csv(data)
    dt["Date_reported"] = pd.to_datetime(dt["Date_reported"])
    return dt
@st.cache
def loadDataV(data):
    dt = pd.read_csv(data)
    return dt

# declare data
covData = loadDataC(covid_data)
vacData = loadDataV(vacine_data)

# select box for desired country
con = st.selectbox("Country", covData["Country"].unique(), index=37)
mainData = covData.loc[covData.Country == con]

st.write(mainData)

# select box for year
year = st.selectbox("Year", [2020, 2021])
yrData = mainData.loc[mainData["Date_reported"].dt.year == year]

# radio buttons for chart
rad = st.radio("Data", ["New_cases", "Cumulative_cases", "New_deaths", "Cumulative_deaths"])

rawData = yrData[["Date_reported", rad]]
rawData["Date_reported"] = rawData["Date_reported"].dt.month

index = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
c1 = rawData.loc[(rawData["Date_reported"] == 1), rad].sum()
c2 = rawData.loc[(rawData["Date_reported"] == 2), rad].sum()
c3 = rawData.loc[(rawData["Date_reported"] == 3), rad].sum()
c4 = rawData.loc[(rawData["Date_reported"] == 4), rad].sum()
c5 = rawData.loc[(rawData["Date_reported"] == 5), rad].sum()
c6 = rawData.loc[(rawData["Date_reported"] == 6), rad].sum()
c7 = rawData.loc[(rawData["Date_reported"] == 7), rad].sum()
c8 = rawData.loc[(rawData["Date_reported"] == 8), rad].sum()
c9 = rawData.loc[(rawData["Date_reported"] == 9), rad].sum()
c10 = rawData.loc[(rawData["Date_reported"] == 10), rad].sum()
c11 = rawData.loc[(rawData["Date_reported"] == 11), rad].sum()
c12 = rawData.loc[(rawData["Date_reported"] == 12), rad].sum()

cols = [c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12]
chartData = pd.DataFrame({
    'Month': index,
    'Data': cols
})

ch = st.select_slider("Chart", ['Histogram', 'Line Graph'])

if ch == 'Histogram':
    # histogram for data
    st.bar_chart(chartData.rename(columns={'Month':'index'}).set_index('index'))
elif ch == 'Line Graph':
    # line chart
    lineData = yrData[["Date_reported", rad]]
    st.line_chart(lineData.rename(columns={'Date_reported':'index'}).set_index('index'))

# machine learning
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

features = ['Date_reported']
X = mainData[features]
y = mainData[rad]
model = RandomForestRegressor(random_state=5)

model.fit(X, y)

st.text("Future Predictions")
st.line_chart(model.predict(X))

# population vacinated
st.text("Vacination Data")
vData = vacData.loc[vacData.COUNTRY == con]
st.write(vData)

