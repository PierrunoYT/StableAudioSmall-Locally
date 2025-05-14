import os
import torch
import torchaudio
import gradio as gr
from einops import rearrange
from stable_audio_tools import get_pretrained_model
from stable_audio_tools.inference.generation import generate_diffusion_cond

# Check for CUDA availability
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Model configuration
MODEL_ID = "stabilityai/stable-audio-open-small"
OUTPUT_DIR = "outputs"

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_model():
    """Load the Stable Audio model"""
    print(f"Loading model {MODEL_ID}...")
    model, model_config = get_pretrained_model(MODEL_ID)
    model = model.to(device)
    sample_rate = model_config["sample_rate"]
    sample_size = model_config["sample_size"]
    print(f"Model loaded. Sample rate: {sample_rate}, Sample size: {sample_size}")
    return model, model_config, sample_rate, sample_size

def generate_audio(
    prompt, 
    duration=5.0, 
    steps=8, 
    cfg_scale=3.0, 
    seed=-1,
    progress=gr.Progress()
):
    """Generate audio from text prompt"""
    # Set seed for reproducibility
    if seed == -1:
        seed = torch.randint(0, 2**32 - 1, (1,)).item()
    torch.manual_seed(seed)
    
    # Set up conditioning
    conditioning = [{
        "prompt": prompt,
        "seconds_total": duration
    }]
    
    progress(0, desc="Initializing...")
    
    # Generate audio
    progress(0.1, desc="Generating audio...")
    output = generate_diffusion_cond(
        model,
        steps=steps,
        conditioning=conditioning,
        sample_size=sample_size,
        cfg_scale=cfg_scale,
        device=device
    )
    
    progress(0.8, desc="Processing audio...")
    
    # Rearrange audio batch to a single sequence
    output = rearrange(output, "b d n -> d (b n)")
    
    # Peak normalize, clip, convert to int16
    output = output.to(torch.float32).div(torch.max(torch.abs(output))).clamp(-1, 1)
    
    # Save to file
    output_path = os.path.join(OUTPUT_DIR, f"output_{seed}.wav")
    torchaudio.save(output_path, output.cpu(), sample_rate)
    
    progress(1.0, desc="Done!")
    return output_path, seed

def create_ui():
    """Create Gradio UI"""
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
        
        # Set up event handlers
        generate_btn.click(
            fn=generate_audio,
            inputs=[prompt, duration, steps, cfg_scale, seed],
            outputs=[output_audio, seed_output]
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
            outputs=[output_audio, seed_output],
            fn=generate_audio
        )
        
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
