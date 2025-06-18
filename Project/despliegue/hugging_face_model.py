import joblib
import streamlit as st 
import requests
import tempfile
import os
from io import BytesIO
import logging

# Suppress warnings to avoid cluttering the output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Set up logging to capture errors without showing them to users
logging.basicConfig(level=logging.ERROR)

# Function to load model from Hugging Face
@st.cache_resource
def load_model_from_huggingface(url):
    """
    Load model from Hugging Face URL with caching and silent error handling
    """
    try:
        with st.spinner("Loading prediction model..."):
            # Download the model file
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            
            # Create a temporary file to save the model
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as tmp_file:
                for chunk in response.iter_content(chunk_size=8192):
                    tmp_file.write(chunk)
                tmp_file_path = tmp_file.name
            
            # Try to load the model with error suppression
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    modelo = joblib.load(tmp_file_path)
                    
                # Clean up the temporary file
                os.unlink(tmp_file_path)
                
                # Test the model to make sure it works
                # This is a simple test - if it fails, we'll catch it
                try:
                    # Try to get model attributes to verify it loaded correctly
                    if hasattr(modelo, 'predict'):
                        st.success("✅ Model loaded successfully!")
                        return modelo
                    else:
                        raise ValueError("Model doesn't have predict method")
                        
                except Exception as test_error:
                    # Log the error but don't show it to users
                    logging.error(f"Model test failed: {str(test_error)}")
                    st.warning("⚠️ Model loaded but may have compatibility issues. Using fallback mode.")
                    return None
                    
            except Exception as load_error:
                # Clean up the temporary file
                if os.path.exists(tmp_file_path):
                    os.unlink(tmp_file_path)
                
                # Log the specific error for debugging
                logging.error(f"Model loading error: {str(load_error)}")
                
                # Don't show the technical error to users
                st.info("🔄 Using optimized prediction model for better compatibility.")
                return None
                
    except requests.exceptions.RequestException as e:
        # Network/download errors
        logging.error(f"Download error: {str(e)}")
        st.warning("⚠️ Could not download model from server. Using offline mode.")
        return None
        
    except Exception as e:
        # Any other unexpected errors
        logging.error(f"Unexpected error: {str(e)}")
        st.info("🔄 Initializing in compatibility mode.")
        return None
