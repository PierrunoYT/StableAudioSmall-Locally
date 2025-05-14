# Stable Audio Web UI

A Gradio-based web interface for Stability AI's Stable Audio model, allowing you to generate audio from text prompts.

## Features

- Generate variable-length (up to 11 seconds) stereo audio at 44.1kHz from text prompts
- Adjust generation parameters like duration, sampling steps, and CFG scale
- Save generated audio files
- Set seeds for reproducible results

## About Stable Audio

Stable Audio is a generative AI model from Stability AI that creates audio from text descriptions. The model used in this project is `stable-audio-open-small`, which comprises:

- An autoencoder that compresses waveforms into a manageable sequence length
- A T5-based text embedding for text conditioning
- A transformer-based diffusion (DiT) model that operates in the latent space of the autoencoder

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/stable-audio-web-ui.git
cd stable-audio-web-ui
```

2. Install the required packages:
```bash
pip install gradio torch torchaudio stable-audio-tools huggingface_hub
```

## Authentication

The Stable Audio model (`stabilityai/stable-audio-open-small`) is a gated model on Hugging Face Hub, which means you need to:

1. Have a Hugging Face account
2. Request access to the model on the [model page](https://huggingface.co/stabilityai/stable-audio-open-small)
3. Generate an access token from your [Hugging Face settings](https://huggingface.co/settings/tokens)

There are three ways to authenticate:

### Option 1: Use the simplified authentication script (recommended)

```bash
python run_with_auth.py --token YOUR_HF_TOKEN
```

This script handles authentication and then launches the main application once it's successful.

### Option 2: Use the authentication UI

1. Run the application without authentication:
```bash
python run.py
```
2. If you're not authenticated, an authentication UI will appear
3. Enter your Hugging Face token and click "Authenticate"
4. After successful authentication, restart the application to load the model properly

### Option 3: Use the command-line argument

```bash
python run.py --token YOUR_HF_TOKEN
```

### Option 4: Set an environment variable

```bash
# On Windows
set HF_TOKEN=YOUR_HF_TOKEN
python run.py

# On Linux/Mac
export HF_TOKEN=YOUR_HF_TOKEN
python run.py
```

## Usage

1. Run the application (with authentication as described above):
```bash
python run.py
```

2. Open your web browser and navigate to `http://127.0.0.1:7860`

3. Enter a text prompt describing the audio you want to generate

4. Adjust parameters as needed:
   - **Duration**: Length of the generated audio (1-11 seconds)
   - **Sampling Steps**: More steps generally produce higher quality but take longer
   - **CFG Scale**: Controls how closely the output follows your prompt
   - **Seed**: Set a specific seed for reproducible results, or -1 for random

5. Click "Generate Audio" and wait for the result

> **Note**: After successful authentication, the application will automatically transition to the main UI. You no longer need to restart the application manually.

### Command-line Options

The application supports several command-line options for debugging and configuration:

```bash
python run.py --help
```

Available options:
- `--share`: Create a publicly shareable link
- `--port PORT`: Port to run the app on (default: 7860)
- `--server-name SERVER_NAME`: Server name (default: 127.0.0.1)
- `--debug`: Enable debug mode with verbose logging
- `--log-level {DEBUG,INFO,WARNING,ERROR,CRITICAL}`: Set logging level (default: INFO)
- `--show-system-info`: Display system information on startup
- `--token TOKEN`: Hugging Face token for authentication

### Debugging Features

The application includes a "Debug Information" section (collapsed by default) that provides:

- System information (platform, Python version, CUDA availability, etc.)
- Current memory usage (RAM and GPU if available)
- Model information (parameters, sample rate, etc.)

Additionally, the UI displays generation time for each audio sample.

## Tips for Better Results

- Be specific about instruments, tempo, and style in your prompts
- Try different CFG scales to control how closely the output follows your prompt
- Use more steps for higher quality (but slower) generation
- Save seeds you like to recreate similar outputs

## Troubleshooting

### Common Issues

- **"NoneType object has no attribute 'pretransform'"**: This error occurs when the model fails to load properly. To fix this:
  1. Ensure you have a valid Hugging Face token with access to the model
  2. Try using the `run_with_auth.py` script instead: `python run_with_auth.py --token YOUR_HF_TOKEN`
  3. If using the authentication UI, make sure to restart the application after successful authentication
  4. Check that you have requested and been granted access to the model on Hugging Face Hub

- **"high is out of bounds for int32"**: This error occurs during random seed generation. It happens in the stable-audio-tools library when trying to generate random numbers outside the int32 range. Our application includes a patch (`seed_patch.py`) that fixes this issue by:
  1. Limiting random seeds to a safe range (max 2^31-1)
  2. Adjusting user-provided seeds to be within int32 bounds
  3. This issue has been reported to the upstream repository: [Issue #195](https://github.com/Stability-AI/stable-audio-tools/issues/195)

- **Authentication Errors**: Make sure you have requested and been granted access to the model on Hugging Face Hub.

- **CUDA Out of Memory**: Try reducing the duration or running on a GPU with more VRAM.

- **Model Loading Errors**: Ensure you have a stable internet connection for downloading the model weights.

## Limitations

- The model is not able to generate realistic vocals
- The model has been trained with English descriptions and will not perform as well in other languages
- The model does not perform equally well for all music styles and cultures
- The model is better at generating sound effects and field recordings than music

## Credits

This project uses:
- [Stable Audio Tools](https://github.com/Stability-AI/stable-audio-tools) by Stability AI
- [Gradio](https://gradio.app/) for the web interface
- [PyTorch](https://pytorch.org/) and [Torchaudio](https://pytorch.org/audio) for audio processing

## License

This project is licensed under the MIT License - see the LICENSE file for details.

The Stable Audio model is licensed under the [Stability AI Community License](https://huggingface.co/stabilityai/stable-audio-open-small/blob/main/LICENSE.md).
