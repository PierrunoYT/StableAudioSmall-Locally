# Stable Audio Web UI Task

## Task Description
Create a Gradio web UI for Stability AI's Stable Audio tools based on the official repository and model.

## Implementation Details

### Files Created
- `app.py`: Main application file containing the Gradio UI and audio generation logic
- `run.py`: Script to run the application with command-line arguments
- `auth.py`: Authentication module for Hugging Face Hub
- `README.md`: Documentation for the project
- `requirements.txt`: List of required packages
- `utils.py`: Utility functions for logging, timing, and system information

### Features Implemented
- Text-to-audio generation using the stable-audio-open-small model
- Authentication with Hugging Face Hub for accessing the gated model
- Multiple authentication methods:
  - Authentication UI
  - Command-line token argument
  - Environment variable
- Adjustable parameters:
  - Duration (1-11 seconds)
  - Sampling steps
  - CFG scale
  - Seed for reproducibility
- Example prompts for quick testing
- Audio playback in the browser
- Saving generated audio files

### Model Used
- Model ID: stabilityai/stable-audio-open-small
- A latent diffusion model based on a transformer architecture
- Generates variable-length (up to 11s) stereo audio at 44.1kHz from text prompts

## How to Run
1. Install dependencies: `pip install -r requirements.txt`
2. Run the application: `python run.py`
3. Optional arguments:
   - `--share`: Create a publicly shareable link
   - `--port`: Specify port (default: 7860)
   - `--server-name`: Specify server name (default: 127.0.0.1)

## Future Improvements
- Add more advanced parameters for audio generation
- Implement batch generation
- Add audio editing capabilities
- Support for model fine-tuning
- Add visualization of the generated audio waveform

## Debugging Features Added
- Comprehensive logging system with configurable log levels
- System information display (hardware, CUDA, memory)
- Memory usage tracking during model loading and generation
- Timing information for all major operations
- Error handling with detailed error messages
- UI-based debugging tools in a collapsible section
- Generation time display for each audio sample
- Command-line options for debugging configuration
