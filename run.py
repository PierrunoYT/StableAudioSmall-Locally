import argparse
from app import create_ui, load_model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Stable Audio Web UI")
    parser.add_argument("--share", action="store_true", help="Create a publicly shareable link")
    parser.add_argument("--port", type=int, default=7860, help="Port to run the app on")
    parser.add_argument("--server-name", type=str, default="127.0.0.1", help="Server name")
    args = parser.parse_args()
    
    # Load model
    global model, model_config, sample_rate, sample_size
    model, model_config, sample_rate, sample_size = load_model()
    
    # Create and launch UI
    app = create_ui()
    app.launch(
        server_name=args.server_name,
        server_port=args.port,
        share=args.share
    )
