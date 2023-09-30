import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="Streamlit Dashboard"
)
# Create a page header
st.header("Baseball statistics for both the American League and National League!")

st.divider()

# Create three columns 
col1, col2= st.columns([1,1])



# inside of the first column
with col1:

    # display a picture
    st.image('images/BaseballAL.png')

    # display the link to that page.
    st.write('<a href="/AmericanLeague"> Check out the statistics of the American League!</a>', unsafe_allow_html=True)

# inside of the second column
with col2:
    # display another picture
    st.image('images/BaseballNL.png')

    # display another link to that page
    st.write('<a href="/NationalLeague">Check out the statistics of the American League!</a>', unsafe_allow_html=True)


