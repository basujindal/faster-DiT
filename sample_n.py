import os
os.environ['PYTHONPATH'] = '/env/python:/content/DiT'
import torch
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from download import find_model
from models import DiT_XL_2
torch.set_grad_enabled(False)
from utils import TimeCuda, Timer
import numpy as np
from PIL import Image
from tqdm import trange, tqdm


def create_npz_from_sample_folder(sample_dir, num=50_000):
    """
    Builds a single .npz file from a folder of .png samples.
    """
    samples = []
    for i in tqdm(range(num), desc="Building .npz file from samples"):
        sample_pil = Image.open(f"{sample_dir}/{i:06d}.png")
        sample_np = np.asarray(sample_pil).astype(np.uint8)
        samples.append(sample_np)
    samples = np.stack(samples)
    assert samples.shape == (num, samples.shape[1], samples.shape[2], 3)
    npz_path = f"{sample_dir}.npz"
    np.savez(npz_path, arr_0=samples)
    print(f"Saved .npz file to {npz_path} [shape={samples.shape}].")
    return npz_path

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
vae_model = "stabilityai/sd-vae-ft-ema" #@param ["stabilityai/sd-vae-ft-mse", "stabilityai/sd-vae-ft-ema"]
latent_size = int(image_size) // 8
sample_folder_dir = "samples"
total = 0
num_samples = 1000
bs = 8


## Load model:
model = DiT_XL_2(input_size=latent_size).to(device)
vae = AutoencoderKL.from_pretrained(vae_model).to(device)

state_dict = find_model(f"DiT-XL-2-{image_size}x{image_size}.pt")
model.load_state_dict(state_dict)

model.eval()
vae.eval()

for i in trange(0, num_samples, bs):

    # Create diffusion object:
    diffusion = create_diffusion(str(num_sampling_steps))

    z = torch.randn(bs, 4, latent_size, latent_size, device=device)
    class_labels = np.random.randint(0, 1000, bs)
    y = torch.tensor(class_labels, device=device)

    # Setup classifier-free guidance:
    z = torch.cat([z, z], 0)
    y_null = torch.tensor([1000] * bs, device=device)
    y = torch.cat([y, y_null], 0)
    model_kwargs = dict(y=y, cfg_scale=cfg_scale)

    # Sample images:
    with torch.no_grad():
        # with Timer():
        with torch.autocast(device_type="cuda"):
            samples = diffusion.ddim_sample_loop(
                model.forward_with_cfg, z.shape, z, clip_denoised=False, 
                model_kwargs=model_kwargs, progress=False, device=device
            )
            samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
            
        samples = vae.decode(samples / 0.18215).sample

    # with Timer():
    samples = torch.clamp(127.5 * samples + 128.0, 0, 255).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()
    for i, sample in enumerate(samples):
        index = i + total
        Image.fromarray(sample).save(f"{sample_folder_dir}/{index:06d}.png")
    total += bs

