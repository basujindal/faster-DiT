import os
os.environ['PYTHONPATH'] = '/env/python:/content/DiT'
import torch
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from models import DiT_XL_2
torch.set_grad_enabled(False)
from utils import TimeCuda, Timer
from accelerate import init_empty_weights
from accelerate.utils import BnbQuantizationConfig, load_and_quantize_model


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

# Create diffusion object:
diffusion = create_diffusion(str(num_sampling_steps))

# Create sampling noise:
n = len(class_labels)

path = "/home/thunder/my_github_clones/faster-DiT/pretrained_models/DiT-XL-2-256x256.pt"


with TimeCuda():

  vae = AutoencoderKL.from_pretrained(vae_model).to(device)

  with init_empty_weights():
    empty_model = DiT_XL_2(input_size=latent_size)

  bnb_quantization_config = BnbQuantizationConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,  # optional
    bnb_4bit_use_double_quant=False,         # optional
    bnb_4bit_quant_type="nf4"               # optional
  )

  model = load_and_quantize_model(
    empty_model,
    weights_location="/home/thunder/my_github_clones/faster-DiT/pretrained_models/DiT-XL-2-256x256.pt",
    bnb_quantization_config=bnb_quantization_config,
    device_map = "auto"
  )

  model.eval()
  vae.eval()

  z = torch.randn(n, 4, latent_size, latent_size, device=device)
  y = torch.tensor(class_labels, device=device)

  # Setup classifier-free guidance:
  z = torch.cat([z, z], 0).to(device)
  y_null = torch.tensor([1000] * n, device=device)
  y = torch.cat([y, y_null], 0).to(device)
  model_kwargs = dict(y=y, cfg_scale=cfg_scale)

  # Sample images:
  with torch.no_grad():
      with Timer("sampling", "Total sampling time = "):
          with torch.autocast(device_type="cuda", dtype = torch.bfloat16):
              # torch.cuda.cudart().cudaProfilerStart()
              samples = diffusion.ddim_sample_loop(
                  model.forward_with_cfg, z.shape, z, clip_denoised=False, 
                  model_kwargs=model_kwargs, progress=True, device=device
              )
              # torch.cuda.cudart().cudaProfilerStop()
              samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
