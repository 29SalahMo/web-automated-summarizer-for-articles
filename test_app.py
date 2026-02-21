"""
Minimal test app to verify Streamlit works
"""
import streamlit as st

st.set_page_config(page_title="Test App", layout="wide")

st.title("✅ Streamlit Test")
st.write("If you see this, Streamlit is working!")

st.success("App loaded successfully!")
