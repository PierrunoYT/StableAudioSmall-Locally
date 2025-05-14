import os
import time
import torch
import torchaudio
import traceback
import gradio as gr
from einops import rearrange
from stable_audio_tools import get_pretrained_model
from stable_audio_tools.inference.generation import generate_diffusion_cond
from utils import logger, Timer, log_memory_usage, timeit

# Check for CUDA availability
device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using device: {device}")

# Model configuration
MODEL_ID = "stabilityai/stable-audio-open-small"
OUTPUT_DIR = "outputs"

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
logger.debug(f"Output directory: {os.path.abspath(OUTPUT_DIR)}")

@timeit
def load_model():
    """Load the Stable Audio model"""
    logger.info(f"Loading model {MODEL_ID}...")

    try:
        with Timer("model_download"):
            logger.debug("Downloading model weights...")
            model, model_config = get_pretrained_model(MODEL_ID)
            logger.debug("Model weights downloaded successfully")

        # Log model configuration details
        logger.debug("Model configuration:")
        for key, value in model_config.items():
            logger.debug(f"  {key}: {value}")

        # Log memory usage before moving model to device
        log_memory_usage("Memory usage before moving model to device")

        with Timer("model_to_device"):
            logger.debug(f"Moving model to {device}...")
            model = model.to(device)
            logger.debug(f"Model moved to {device}")

        # Log memory usage after moving model to device
        log_memory_usage("Memory usage after moving model to device")

        sample_rate = model_config["sample_rate"]
        sample_size = model_config["sample_size"]

        # Log model parameters count
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Model loaded successfully:")
        logger.info(f"  Sample rate: {sample_rate} Hz")
        logger.info(f"  Sample size: {sample_size}")
        logger.info(f"  Total parameters: {total_params:,}")
        logger.info(f"  Trainable parameters: {trainable_params:,}")

        return model, model_config, sample_rate, sample_size

    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        logger.error(traceback.format_exc())
        raise

@timeit
def generate_audio(
    prompt,
    duration=5.0,
    steps=8,
    cfg_scale=3.0,
    seed=-1,
    progress=gr.Progress()
):
    """Generate audio from text prompt"""
    logger.info(f"Generating audio with parameters:")
    logger.info(f"  Prompt: '{prompt}'")
    logger.info(f"  Duration: {duration} seconds")
    logger.info(f"  Steps: {steps}")
    logger.info(f"  CFG Scale: {cfg_scale}")

    try:
        # Set seed for reproducibility
        if seed == -1:
            seed = torch.randint(0, 2**32 - 1, (1,)).item()
            logger.debug(f"Generated random seed: {seed}")
        else:
            logger.debug(f"Using provided seed: {seed}")

        torch.manual_seed(seed)

        # Set up conditioning
        conditioning = [{
            "prompt": prompt,
            "seconds_total": duration
        }]

        progress(0, desc="Initializing...")
        logger.debug("Initializing generation process")

        # Log memory usage before generation
        log_memory_usage("Memory usage before generation")

        # Generate audio
        progress(0.1, desc="Generating audio...")
        logger.debug(f"Starting diffusion process with {steps} steps")

        with Timer("diffusion_generation"):
            output = generate_diffusion_cond(
                model,
                steps=steps,
                conditioning=conditioning,
                sample_size=sample_size,
                cfg_scale=cfg_scale,
                device=device
            )
            logger.debug(f"Diffusion completed, output shape: {output.shape}")

        progress(0.8, desc="Processing audio...")
        logger.debug("Post-processing audio")

        # Rearrange audio batch to a single sequence
        with Timer("audio_rearrange"):
            output = rearrange(output, "b d n -> d (b n)")
            logger.debug(f"Audio rearranged, new shape: {output.shape}")

        # Peak normalize, clip, convert to int16
        with Timer("audio_normalize"):
            max_value = torch.max(torch.abs(output))
            logger.debug(f"Peak value before normalization: {max_value.item():.4f}")
            output = output.to(torch.float32).div(max_value).clamp(-1, 1)
            logger.debug("Audio normalized and clamped")

        # Save to file
        output_path = os.path.join(OUTPUT_DIR, f"output_{seed}.wav")
        with Timer("save_audio"):
            logger.debug(f"Saving audio to {output_path}")
            torchaudio.save(output_path, output.cpu(), sample_rate)
            logger.debug(f"Audio saved successfully, file size: {os.path.getsize(output_path) / 1024:.2f} KB")

        # Log memory usage after generation
        log_memory_usage("Memory usage after generation")

        progress(1.0, desc="Done!")
        logger.info(f"Audio generation completed successfully with seed {seed}")
        return output_path, seed

    except Exception as e:
        logger.error(f"Error generating audio: {str(e)}")
        logger.error(traceback.format_exc())
        raise

@timeit
def create_ui():
    """Create Gradio UI"""
    logger.debug("Creating Gradio UI")

    with gr.Blocks(title="Stable Audio Generator") as app:
        gr.Markdown("# 🎵 Stable Audio Generator")
        gr.Markdown("Generate audio from text prompts using Stability AI's Stable Audio model.")

        with gr.Row():
            with gr.Column():
                prompt = gr.Textbox(
                    label="Prompt",
                    placeholder="Enter a description of the audio you want to generate...",
                    value="128 BPM tech house drum loop with deep bass"
                )

                with gr.Row():
                    duration = gr.Slider(
                        minimum=1.0,
                        maximum=11.0,
                        value=5.0,
                        step=0.5,
                        label="Duration (seconds)"
                    )
                    steps = gr.Slider(
                        minimum=1,
                        maximum=50,
                        value=8,
                        step=1,
                        label="Sampling Steps"
                    )

                with gr.Row():
                    cfg_scale = gr.Slider(
                        minimum=1.0,
                        maximum=10.0,
                        value=3.0,
                        step=0.1,
                        label="CFG Scale"
                    )
                    seed = gr.Number(
                        label="Seed (-1 for random)",
                        value=-1,
                        precision=0
                    )

                generate_btn = gr.Button("🎵 Generate Audio", variant="primary")

            with gr.Column():
                output_audio = gr.Audio(label="Generated Audio", type="filepath")
                seed_output = gr.Number(label="Seed Used", precision=0)

                # Add generation time display
                generation_time = gr.Textbox(label="Generation Time", value="", interactive=False)

        # Set up event handlers
        def wrapped_generate_audio(*args):
            start_time = time.time()
            try:
                output_path, seed = generate_audio(*args)
                end_time = time.time()
                elapsed = end_time - start_time
                return output_path, seed, f"{elapsed:.2f} seconds"
            except Exception as e:
                logger.error(f"Error in wrapped_generate_audio: {str(e)}")
                logger.error(traceback.format_exc())
                end_time = time.time()
                elapsed = end_time - start_time
                return None, -1, f"Error after {elapsed:.2f} seconds: {str(e)}"

        generate_btn.click(
            fn=wrapped_generate_audio,
            inputs=[prompt, duration, steps, cfg_scale, seed],
            outputs=[output_audio, seed_output, generation_time]
        )

        # Examples
        gr.Examples(
            examples=[
                ["128 BPM tech house drum loop with deep bass", 5.0, 8, 3.0, -1],
                ["Ambient synth pad with reverb", 8.0, 12, 3.0, -1],
                ["Acoustic guitar playing a folk melody", 6.0, 10, 3.0, -1],
                ["Birds chirping in a forest", 4.0, 8, 3.0, -1],
                ["Cinematic orchestral music with strings and brass", 10.0, 15, 3.0, -1]
            ],
            inputs=[prompt, duration, steps, cfg_scale, seed],
            outputs=[output_audio, seed_output, generation_time],
            fn=wrapped_generate_audio
        )

        # Add debug info accordion
        with gr.Accordion("Debug Information", open=False):
            system_info_btn = gr.Button("Show System Info")
            system_info_output = gr.Textbox(label="System Information", interactive=False, lines=10)

            memory_usage_btn = gr.Button("Show Memory Usage")
            memory_usage_output = gr.Textbox(label="Memory Usage", interactive=False, lines=6)

            model_info_btn = gr.Button("Show Model Info")
            model_info_output = gr.Textbox(label="Model Information", interactive=False, lines=8)

        # Debug button handlers
        def get_system_info_text():
            from utils import get_system_info
            info = get_system_info()
            return "\n".join([f"{key}: {value}" for key, value in info.items()])

        def get_memory_usage_text():
            from utils import get_memory_usage
            usage = get_memory_usage()
            return "\n".join([f"{key}: {value}" for key, value in usage.items()])

        def get_model_info_text():
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

            info = {
                "Model ID": MODEL_ID,
                "Sample rate": f"{sample_rate} Hz",
                "Sample size": sample_size,
                "Total parameters": f"{total_params:,}",
                "Trainable parameters": f"{trainable_params:,}",
                "Device": device
            }

            return "\n".join([f"{key}: {value}" for key, value in info.items()])

        system_info_btn.click(fn=get_system_info_text, inputs=[], outputs=[system_info_output])
        memory_usage_btn.click(fn=get_memory_usage_text, inputs=[], outputs=[memory_usage_output])
        model_info_btn.click(fn=get_model_info_text, inputs=[], outputs=[model_info_output])

        gr.Markdown("""
        ## Tips for better results:
        - Be specific about the instruments, tempo, and style
        - Try different CFG scales to control how closely the output follows your prompt
        - Use more steps for higher quality (but slower) generation
        - Save seeds you like to recreate similar outputs
        """)

    return app

# Load model globally
print("Initializing Stable Audio...")
model, model_config, sample_rate, sample_size = load_model()

# Create and launch the UI
if __name__ == "__main__":
    app = create_ui()
    app.launch(share=False)
