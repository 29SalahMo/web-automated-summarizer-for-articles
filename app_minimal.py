"""
Minimal test version - if this works, we know Streamlit is fine
"""
import streamlit as st

st.set_page_config(page_title="Test", layout="wide")
st.title("✅ Streamlit Works!")
st.write("If you see this, Streamlit is working correctly.")
st.success("App loaded successfully!")
