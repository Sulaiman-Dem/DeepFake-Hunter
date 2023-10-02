import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px

#Page title
st.set_page_config(
    page_title="National League"
)

#Loading in data
df = pd.read_csv('data/baseball.csv')
Aleague = df[df['League']=='AL'].copy()
Nleague = df[df['League']=='NL'].copy()

#Creating a separate dataframe to graph the Year and average RS for NL
NL_y_rs = Nleague.groupby('Year')['RS'].mean().reset_index()
fig = px.bar(NL_y_rs,
        x='Year',
        y='RS', 
        title="Average Runs Scored per Year.",
        labels={ "RS": "Average Runs Scored"} 
        )

# display graph
st.plotly_chart(fig, use_container_width=True)