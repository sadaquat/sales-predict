import os
from dotenv import load_dotenv

load_dotenv()

def get_secret(key, default=None):
    """Get secret from Streamlit secrets (cloud) or .env file (local)."""
    # Try Streamlit secrets first
    try:
        import streamlit as st
        val = st.secrets.get(key)
        if val is not None:
            return val
    except Exception:
        pass
    
    # Fall back to environment variable
    val = os.getenv(key, default)
    return val