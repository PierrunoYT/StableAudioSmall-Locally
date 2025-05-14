import argparse
import os
# Import our patch to fix the int32 seed overflow issue
import seed_patch
from app import create_ui, load_model, create_auth_ui
from utils import set_log_level, print_system_info, log_memory_usage, logger

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Stable Audio Web UI")
    parser.add_argument("--share", action="store_true", help="Create a publicly shareable link")
    parser.add_argument("--port", type=int, default=7860, help="Port to run the app on")
    parser.add_argument("--server-name", type=str, default="127.0.0.1", help="Server name")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--log-level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        help="Set logging level")
    parser.add_argument("--show-system-info", action="store_true", help="Display system information on startup")
    parser.add_argument("--token", type=str, help="Hugging Face token for authentication")
    args = parser.parse_args()

    # Configure logging
    if args.debug:
        set_log_level("DEBUG")
    else:
        set_log_level(args.log_level)

    # Display system information if requested
    if args.show_system_info or args.debug:
        print_system_info()

    logger.info("Starting Stable Audio Web UI")

    # Set token from command line to environment variable if provided
    if args.token:
        os.environ["HF_TOKEN"] = args.token
        logger.info("Using token from command line")

    # Load model
    # Note: These are already defined as global in app.py
    logger.info("Initializing Stable Audio...")
    try:
        # Try to load the model with authentication
        from app import model, model_config, sample_rate, sample_size
        # Update the global variables in the app module
        import app as app_module
        app_module.model, app_module.model_config, app_module.sample_rate, app_module.sample_size = load_model()

        # Log memory usage after model loading
        if args.debug:
            log_memory_usage("Memory usage after model loading")

        # Create and launch UI
        logger.info("Creating UI...")
        app = create_ui()

        logger.info(f"Launching server on {args.server_name}:{args.port} (share={args.share})")
        app.launch(
            server_name=args.server_name,
            server_port=args.port,
            share=args.share
        )
    except Exception as e:
        logger.error(f"Failed to initialize model: {str(e)}")
        # If model loading fails, create a UI with authentication
        create_auth_ui()
