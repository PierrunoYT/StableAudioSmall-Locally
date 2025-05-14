import os
import logging
from huggingface_hub import login, whoami, HfApi
from utils import logger

def is_authenticated():
    """Check if the user is authenticated with Hugging Face.
    
    Returns:
        bool: True if authenticated, False otherwise
    """
    try:
        # Try to get user info to check if authenticated
        user_info = whoami()
        logger.info(f"Authenticated as {user_info['name']} ({user_info['email']})")
        return True
    except Exception as e:
        logger.debug(f"Not authenticated: {str(e)}")
        return False

def login_with_token(token):
    """Login to Hugging Face with a token.
    
    Args:
        token (str): Hugging Face token
        
    Returns:
        bool: True if login successful, False otherwise
    """
    try:
        if not token or token.strip() == "":
            logger.error("Token is empty")
            return False
            
        # Try to login with the provided token
        login(token=token, write_permission=True)  # Save the token to disk cache
        
        # Verify authentication was successful
        if is_authenticated():
            logger.info("Successfully authenticated with Hugging Face")
            return True
        else:
            logger.error("Authentication failed")
            return False
    except Exception as e:
        logger.error(f"Error during authentication: {str(e)}")
        return False

def login_with_env_token():
    """Try to login using the HF_TOKEN environment variable.
    
    Returns:
        bool: True if login successful, False otherwise
    """
    token = os.environ.get("HF_TOKEN")
    if token:
        logger.info("Found HF_TOKEN environment variable, attempting to authenticate")
        return login_with_token(token)
    else:
        logger.debug("No HF_TOKEN environment variable found")
        return False

def check_model_access(model_id):
    """Check if the authenticated user has access to the specified model.
    
    Args:
        model_id (str): The model ID to check access for
        
    Returns:
        bool: True if the user has access, False otherwise
    """
    try:
        api = HfApi()
        # Try to get model info to check access
        model_info = api.model_info(model_id)
        logger.info(f"User has access to model {model_id}")
        return True
    except Exception as e:
        logger.error(f"User does not have access to model {model_id}: {str(e)}")
        return False
