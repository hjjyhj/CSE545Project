import os
from python_dotenv import load_dotenv

def load_environment_variables():
    """
    Load environment variables from .env file
    
    Returns:
        dict: Dictionary containing environment variables
    """
    # Load .env file
    load_dotenv()
    
    # Get API key for OpenRouter
    openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
    
    if not openrouter_api_key:
        raise ValueError("OPENROUTER_API_KEY is not set in .env file")
    
    # Get site URL and name for OpenRouter rankings (optional)
    site_url = os.getenv("SITE_URL", "http://localhost:3000")
    site_name = os.getenv("SITE_NAME", "CSE545Project")
    
    return {
        "openrouter_api_key": openrouter_api_key,
        "site_url": site_url,
        "site_name": site_name
    } 