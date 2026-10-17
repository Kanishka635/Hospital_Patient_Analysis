
import streamlit as st

st.set_page_config(page_title="Dashboard Test", layout="wide")

st.title("Hospital Patient Arrival Analysis")
st.success("Great! Your dashboard is displaying correctly.")

col1, col2, col3 = st.columns(3)
col1.metric("Total Arrivals", "1,250")
col2.metric("Average Per Day", "42")
col3.metric("Highest Arrivals", "68")

st.line_chart([25, 32, 28, 45, 38, 52, 42])
