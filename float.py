import os
os.environ['PYTHONPATH'] = '/env/python:/content/DiT'
import torch
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from download import find_model
from models import DiT_XL_2
torch.set_grad_enabled(False)
from utils import TimeCuda, Timer

device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cpu":
    print("GPU not found. Using CPU instead.")


image_size = 256 #@param [256, 512]
vae_model = "stabilityai/sd-vae-ft-ema" #@param ["stabilityai/sd-vae-ft-mse", "stabilityai/sd-vae-ft-ema"]
latent_size = int(image_size) // 8
seed = 0 #@param {type:"number"}
torch.manual_seed(seed)
num_sampling_steps = 50 #@param {type:"slider", min:0, max:1000, step:1}
cfg_scale = 4 #@param {type:"slider", min:1, max:10, step:0.1}
class_labels = 207, 360, 387, 974, 88, 979, 417, 279 #@param {type:"raw"}
samples_per_row = 4 #@param {type:"number"}
vae_model = "stabilityai/sd-vae-ft-ema" #@param ["stabilityai/sd-vae-ft-mse", "stabilityai/sd-vae-ft-ema"]
latent_size = int(image_size) // 8

# Create diffusion object:
diffusion = create_diffusion(str(num_sampling_steps))

# Create sampling noise:
n = len(class_labels)

with TimeCuda():

    ## Load model:
    model = DiT_XL_2(input_size=latent_size).to(device)
    vae = AutoencoderKL.from_pretrained(vae_model).to(device)

    state_dict = find_model(f"DiT-XL-2-{image_size}x{image_size}.pt")
    model.load_state_dict(state_dict)

    model.eval()
    vae.eval()

    z = torch.randn(n, 4, latent_size, latent_size, device=device)
    y = torch.tensor(class_labels, device=device)

    # Setup classifier-free guidance:
    z = torch.cat([z, z], 0)
    y_null = torch.tensor([1000] * n, device=device)
    y = torch.cat([y, y_null], 0)
    model_kwargs = dict(y=y, cfg_scale=cfg_scale)

    # Sample images:
    with torch.no_grad():
        with Timer("Sampling","Total sampling time = "):
            with torch.autocast(device_type="cuda"):
                samples = diffusion.ddim_sample_loop(
                    model.forward_with_cfg, z.shape, z, clip_denoised=False, 
                    model_kwargs=model_kwargs, progress=True, device=device
                )
                samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
                samples = vae.decode(samples / 0.18215).sample
