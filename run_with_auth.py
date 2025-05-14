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
            
            # Run the app with the token already set in environment
            cmd = [sys.executable, "run.py"]
            if args.port:
                cmd.extend(["--port", str(args.port)])
            if args.share:
                cmd.append("--share")
            if args.debug:
                cmd.append("--debug")
                
            # Execute the app
            try:
                subprocess.run(cmd)
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