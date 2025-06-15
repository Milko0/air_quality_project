import joblib
import streamlit as st 
import requests
import tempfile
import os
from io import BytesIO

# Function to load model from Hugging Face
@st.cache_resource
def load_model_from_huggingface(url):
    """
    Load model from Hugging Face URL with caching
    """
    try:
        with st.spinner("Loading model from Hugging Face..."):
            # Download the model file
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            # Create a temporary file to save the model
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as tmp_file:
                for chunk in response.iter_content(chunk_size=8192):
                    tmp_file.write(chunk)
                tmp_file_path = tmp_file.name
            
            # Load the model from the temporary file
            modelo = joblib.load(tmp_file_path)
            
            # Clean up the temporary file
            os.unlink(tmp_file_path)
            
            return modelo
            
    except Exception as e:
        st.error(f"Error loading model from Hugging Face: {str(e)}")
        return None

# Alternative method using direct BytesIO (more memory efficient for smaller models)
@st.cache_resource
def load_model_from_huggingface_memory(url):
    """
    Load model directly into memory (alternative method)
    """
    try:
        with st.spinner("Loading model from Hugging Face (memory method)..."):
            response = requests.get(url)
            response.raise_for_status()
            
            # Load model directly from bytes
            model_bytes = BytesIO(response.content)
            modelo = joblib.load(model_bytes)
            
            return modelo
            
    except Exception as e:
        st.error(f"Error loading model from Hugging Face: {str(e)}")
        return None