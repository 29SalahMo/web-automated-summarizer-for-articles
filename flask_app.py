# This file redirects to streamlit_app.py for Hugging Face Spaces
# Hugging Face Spaces looks for app.py by default, so we import the Streamlit app here

import streamlit_app

# Run the Streamlit app
if __name__ == "__main__":
    streamlit_app.main()
