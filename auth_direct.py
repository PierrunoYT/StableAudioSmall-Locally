import os
import logging
import argparse
from huggingface_hub import login, whoami, HfApi

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("auth")

def login_with_token(token, write_to_env=True):
    """Login to Hugging Face with a token.
    
    Args:
        token (str): Hugging Face token
        write_to_env (bool): Whether to write the token to the environment variable
    """
    try:
        if not token or token.strip() == "":
            logger.error("Token is empty")
            return False
            
        # Try to login with the provided token
        logger.info("Authenticating with HuggingFace...")
        login(token=token, write_permission=True)
        
        # Verify authentication was successful
        try:
            user_info = whoami()
            logger.info(f"Successfully authenticated as {user_info['name']} ({user_info['email']})")
            
            # Save token to environment variable for future use
            if write_to_env:
                os.environ["HF_TOKEN"] = token
                logger.info("Token saved to environment variable HF_TOKEN")
                
            return True
        except Exception as e:
            logger.error(f"Authentication failed: {str(e)}")
            return False
            
    except Exception as e:
        logger.error(f"Error during authentication: {str(e)}")
        return False

def check_model_access(model_id="stabilityai/stable-audio-open-small"):
    """Check if the authenticated user has access to the model.
    
    Args:
        model_id (str): The HuggingFace model ID
        
    Returns:
        bool: True if the user has access, False otherwise
    """
    try:
        logger.info(f"Checking access to model {model_id}...")
        api = HfApi()
        model_info = api.model_info(model_id)
        logger.info(f"✅ Access confirmed to model {model_id}")
        return True
    except Exception as e:
        logger.error(f"❌ No access to model {model_id}: {str(e)}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Authenticate with HuggingFace")
    parser.add_argument("--token", type=str, help="HuggingFace access token")
    parser.add_argument("--model", type=str, default="stabilityai/stable-audio-open-small", 
                        help="Model ID to check access for")
    args = parser.parse_args()

    token = args.token or os.environ.get("HF_TOKEN")
    
    if not token:
        token = input("Enter your HuggingFace access token: ")
    
    if login_with_token(token):
        check_model_access(args.model)
        print("\nTo use this token in your application, restart it or set the environment variable:")
        print(f"export HF_TOKEN={token}")
    else:
        print("\n❌ Authentication failed. Please try again with a valid token.")
        print("You can generate a token at: https://huggingface.co/settings/tokens")