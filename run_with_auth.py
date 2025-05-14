#!/usr/bin/env python3
"""
Simple script to authenticate with Hugging Face and then run the Stable Audio app.
This bypasses the problematic authentication flow in the main app.
"""

import os
import sys
import subprocess
import argparse
from auth_direct import login_with_token, check_model_access

def main():
    parser = argparse.ArgumentParser(description="Authenticate with HuggingFace and run Stable Audio")
    parser.add_argument("--token", type=str, help="HuggingFace access token")
    parser.add_argument("--port", type=int, default=7860, help="Port to run the app on")
    parser.add_argument("--share", action="store_true", help="Create a publicly shareable link")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    args = parser.parse_args()

    # Get token from args or environment
    token = args.token or os.environ.get("HF_TOKEN")
    
    if not token:
        token = input("Enter your HuggingFace access token: ")
    
    # Authenticate with the token
    if login_with_token(token):
        # Check model access
        if check_model_access("stabilityai/stable-audio-open-small"):
            print("\n✅ Authentication successful! Starting Stable Audio...\n")
            
            # Instead of subprocess, directly import and run the app
            # This ensures we use the same Python interpreter and global variables
            print("\nStarting Stable Audio application...")
            
            # Import the necessary modules
            # First import the seed patch to fix the int32 overflow issue
            import seed_patch
            import app
            from app import create_ui, load_model
            import sys
            
            # Update the global model variables directly
            try:
                print("Loading model...")
                app.model, app.model_config, app.sample_rate, app.sample_size = load_model()
                
                # Create and launch UI
                print("Creating UI...")
                ui = create_ui()
                
                # Launch with the appropriate parameters
                share = args.share
                port = args.port or 7860
                debug = args.debug
                
                print(f"Launching server on port {port} (share={share})...")
                ui.launch(
                    server_port=port,
                    share=share,
                    debug=debug
                )
            except KeyboardInterrupt:
                print("\nApplication stopped by user.")
        else:
            print("\n❌ You don't have access to the model. Request access at:")
            print("https://huggingface.co/stabilityai/stable-audio-open-small")
    else:
        print("\n❌ Authentication failed. Please try again with a valid token.")
        print("You can generate a token at: https://huggingface.co/settings/tokens")

if __name__ == "__main__":
    main()