import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px

#Page title
st.set_page_config(
    page_title="American League"
)

#Loading in data
df = pd.read_csv('data/baseball.csv')
Aleague = df[df['League']=='AL'].copy()
Nleague = df[df['League']=='NL'].copy()

# create a list of all the state names
team_list = sorted(Aleague['Team'].unique())
    
# create a multi select button
selected_teams = st.multiselect(
    'Select which teams to compare.',
    team_list,
    default=['ANA'])

team_df = Aleague[Aleague['Team'].isin(selected_teams)].groupby('Team')['RS'].mean().reset_index().copy()

#Creating bar graph
fig = px.bar(team_df,
        x='Team',
        y='RS', 
        text_auto='.2s',
        title="Average Runs Scored per Year by Team.",
        labels={ "RS": "Average Runs Scored per Year",
                "Team": "Teams"} 
        )

# changing the grid axes
fig.update_xaxes(showgrid=False, gridwidth=1, gridcolor='Gray')
fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='Gray')

# displays 1st graph
st.plotly_chart(fig, use_container_width=True)

#Creating a separate dataframe to graph the Year and average RS for AL
AL_y_rs = Aleague.groupby('Year')['RS'].mean().reset_index()
fig = px.bar(AL_y_rs,
        x='Year',
        y='RS', 
        title="Average Runs Scored per Year for all Teams.",
        labels={ "RS": "Average Runs Scored"} 
        )

# displays 2nd graph
st.plotly_chart(fig, use_container_width=True)