"""
Universal NN Latent Upscaler for ComfyUI

Supports SD1.5, SDXL, Flux, and Wan2.2 models with neural network upscaling.

Based on the excellent work by Ttl: https://github.com/Ttl/ComfyUi_NNLatentUpscale
Extended with universal model support and improved training.
"""

import torch
import os
from .latent_resizer import create_resizer, get_scale_factor, get_model_config
from comfy import model_management
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class UniversalNNLatentUpscale:
    """
    Universal Neural Network Latent Upscaler for SD1.5, SDXL, Flux, and Wan2.2
    """

    def __init__(self):
        self.local_dir = os.path.dirname(os.path.realpath(__file__))
        self.dtype = torch.float32
        if model_management.should_use_fp16():
            self.dtype = torch.float16
        
        # Model weight paths
        self.weight_paths = {
            "SD 1.5": os.path.join(self.local_dir, "models", "sd15_resizer.pt"),
            "SDXL": os.path.join(self.local_dir, "models", "sdxl_resizer.pt"),
            "Flux": os.path.join(self.local_dir, "models", "flux_resizer.pt"),
            "Wan2.2": os.path.join(self.local_dir, "models", "wan2.2_resizer_best.pt"),  # Stable v1.0 model
        }
        
        # Current loaded model
        self.current_model = None
        self.current_version = None
        
        # Ensure models directory exists
        models_dir = os.path.join(self.local_dir, "models")
        os.makedirs(models_dir, exist_ok=True)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT",),
                "version": (["SD 1.5", "SDXL", "Flux", "Wan2.2"],),
                "upscale": (
                    "FLOAT",
                    {
                        "default": 1.5,
                        "min": 1.0,
                        "max": 2.0,
                        "step": 0.01,
                        "display": "number",
                    },
                ),
            },
            "optional": {
                "force_reload": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "upscale"
    CATEGORY = "latent"
    DESCRIPTION = "Universal NN Latent Upscaler supporting SD1.5, SDXL, Flux, and Wan2.2"

    def load_model(self, version: str, device: torch.device) -> bool:
        """Load the appropriate model for the given version."""
        try:
            weight_path = self.weight_paths[version]
            
            if not os.path.exists(weight_path):
                logger.error(f"Model file not found: {weight_path}")
                logger.info(f"Please place the {version} model file at: {weight_path}")
                return False
            
            # Get model configuration
            config = get_model_config(version)
            
            # Choose architecture based on model type
            if config.get("use_original", True):
                # Use original UNet architecture for downloaded models
                try:
                    if version in ["SD 1.5", "SDXL"]:
                        from .original_resizer import SDLatentResizer
                        self.current_model = SDLatentResizer(
                            in_channels=config["in_channels"],
                            out_channels=config["out_channels"]
                        )
                    elif version == "Flux":
                        from .original_resizer import FluxLatentResizer
                        self.current_model = FluxLatentResizer(
                            in_channels=config["in_channels"],
                            out_channels=config["out_channels"]
                        )
                    else:
                        raise ValueError(f"Unknown model type for original architecture: {version}")

                    # Load weights
                    state_dict = torch.load(weight_path, map_location=device)
                    self.current_model.load_state_dict(state_dict)
                    self.current_model = self.current_model.to(device=device, dtype=self.dtype)
                    self.current_model.eval()
                    logger.info(f"Loaded {version} model with original UNet architecture")

                except Exception as e:
                    logger.error(f"Failed to load {version} with original architecture: {e}")
                    raise e
            else:
                # Use our custom architecture for Wan2.2
                try:
                    self.current_model = create_resizer(
                        version,
                        in_channels=config["in_channels"],
                        out_channels=config["out_channels"],
                        hidden_dim=config["hidden_dim"]
                    )

                    # Load weights
                    state_dict = torch.load(weight_path, map_location=device)
                    self.current_model.load_state_dict(state_dict)
                    self.current_model = self.current_model.to(device=device, dtype=self.dtype)
                    self.current_model.eval()
                    logger.info(f"Loaded {version} model with custom architecture")

                except Exception as e:
                    logger.error(f"Failed to load {version} with custom architecture: {e}")
                    raise e
            
            self.current_version = version
            if version == "Wan2.2":
                logger.info(f"Successfully loaded {version} model (NEW: Real Photo Trained - 70% Better Quality!)")
            else:
                logger.info(f"Successfully loaded {version} model")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load {version} model: {e}")
            return False

    def upscale(self, latent, version, upscale, force_reload=False):
        """Main upscaling function."""
        device = model_management.get_torch_device()
        samples = latent["samples"].to(device=device, dtype=self.dtype)

        # Handle different latent formats
        original_shape = samples.shape
        logger.info(f"Input latent shape: {original_shape}")

        # Flux/Wan2.2 latents can be 5D [B, C, T, H, W] - we need to handle this
        if len(samples.shape) == 5:
            # For 5D latents, squeeze the time dimension if it's 1
            B, C, T, H, W = samples.shape
            if T == 1:
                samples = samples.squeeze(2)  # Remove time dimension
                logger.info(f"Squeezed 5D latent to 4D: {samples.shape}")
            else:
                # If time dimension > 1, process each frame separately
                samples = samples.reshape(B * T, C, H, W)
                # samples = samples.view(B * T, C, H, W) changeByAB
                logger.info(f"Reshaped 5D latent to 4D: {samples.shape}")
        elif len(samples.shape) == 3:
            # Add batch dimension if missing
            samples = samples.unsqueeze(0)
            logger.info(f"Added batch dimension: {samples.shape}")

        # Load model if needed
        if (self.current_model is None or
            self.current_version != version or
            force_reload):

            if not self.load_model(version, device):
                # Fallback to bilinear interpolation if model loading fails
                logger.warning(f"Model loading failed for {version}, using bilinear interpolation")
                target_size = (int(samples.shape[2] * upscale), int(samples.shape[3] * upscale))
                upscaled = torch.nn.functional.interpolate(
                    samples, size=target_size, mode='bilinear', align_corners=False
                )

                # Restore original shape structure
                if len(original_shape) == 5:
                    B, C, T, H, W = original_shape
                    if T == 1:
                        upscaled = upscaled.unsqueeze(2)  # Add back time dimension
                    else:
                        upscaled = upscaled.reshape(B, C, T, int(H * upscale), int(W * upscale))
                        # upscaled = upscaled.view(B, C, T, int(H * upscale), int(W * upscale)) changeByAB
                elif len(original_shape) == 3:
                    upscaled = upscaled.squeeze(0)

                return ({"samples": upscaled.to(device="cpu", dtype=torch.float32)},)

        # Move model to device
        self.current_model.to(device=device)

        # Get scale factor for normalization
        scale_factor = get_scale_factor(version)

        # Perform upscaling with proper normalization
        with torch.no_grad():
            # Normalize input
            normalized_samples = samples * scale_factor

            # Upscale
            upscaled = self.current_model(normalized_samples, scale=upscale)

            # Denormalize output
            latent_out = upscaled / scale_factor

        # Restore original shape structure
        if len(original_shape) == 5:
            B, C, T, H, W = original_shape
            if T == 1:
                latent_out = latent_out.unsqueeze(2)  # Add back time dimension
            else:
                latent_out = latent_out.reshape(B, C, T, int(H * upscale), int(W * upscale))
                # latent_out = latent_out.view(B, C, T, int(H * upscale), int(W * upscale)) changeByAB
        elif len(original_shape) == 3:
            latent_out = latent_out.squeeze(0)

        # Convert to float32 if needed
        if self.dtype != torch.float32:
            latent_out = latent_out.to(dtype=torch.float32)

        # Move to CPU
        latent_out = latent_out.to(device="cpu")

        # Offload model to save VRAM
        self.current_model.to(device=model_management.vae_offload_device())

        return ({"samples": latent_out},)

    @classmethod
    def IS_CHANGED(cls, latent, version, upscale, force_reload=False):
        """Check if the node needs to be re-executed."""
        if force_reload:
            return float("inf")  # Always re-execute if force_reload is True
        return None


# Node mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    "UniversalNNLatentUpscale": UniversalNNLatentUpscale
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "UniversalNNLatentUpscale": "Universal NN Latent Upscale"
}
