"""
Patch for stable-audio-tools library to fix the int32 seed overflow issue.
This module monkey-patches the generate_diffusion_cond function to use a safe seed range.
"""

import numpy as np
import logging
from stable_audio_tools.inference.generation import generate_diffusion_cond as original_generate_diffusion_cond

logger = logging.getLogger("stable-audio")

def patched_generate_diffusion_cond(model, steps=10, batch_size=1, conditioning=None, 
                                   sample_size=524288, cfg_scale=3.0, seed=-1, device="cuda"):
    """
    Patched version of generate_diffusion_cond that fixes the int32 overflow issue.
    
    This wrapper ensures that random seeds are within the bounds of int32,
    preventing the "high is out of bounds for int32" error.
    """
    # Fix the seed value to be within int32 bounds
    if seed == -1:
        # Use a safe maximum value for int32
        seed = np.random.randint(0, 2**31 - 1)
        logger.debug(f"Generated safe random seed: {seed}")
    else:
        # Ensure provided seed is within int32 bounds
        seed = int(seed) % (2**31 - 1)
        logger.debug(f"Adjusted seed to safe value: {seed}")
    
    # Call the original function with the safe seed
    return original_generate_diffusion_cond(
        model=model, 
        steps=steps, 
        batch_size=batch_size, 
        conditioning=conditioning, 
        sample_size=sample_size, 
        cfg_scale=cfg_scale, 
        seed=seed, 
        device=device
    )

# Monkey patch the original function
import stable_audio_tools.inference.generation
stable_audio_tools.inference.generation.generate_diffusion_cond = patched_generate_diffusion_cond

logger.info("Patched stable_audio_tools.inference.generation.generate_diffusion_cond to fix int32 seed overflow")