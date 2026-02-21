# Entry point for Hugging Face Spaces (Streamlit)
# Hugging Face Spaces looks for app.py by default, so we import the Streamlit app here

# Simply import and run the Streamlit app
import streamlit_app

# Call main() to run the app
if __name__ == "__main__":
    streamlit_app.main()
else:
    # When imported by Streamlit, ensure main is available
    streamlit_app.main()
