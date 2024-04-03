# Goal

4 bit (NF or INT4 or FP4) storage and 8bit computation using int8 matrix multiplication of Diffusion Transformer model (DiT)

## Plan

- Start with simple quantization for storage and see the impact on inference latency and quality. (Since model is small, no outliers are there)
- Use bnb to check int8 and int4 quantization.
- Use bnb to check int8 matrix multiplication and use their kernels
- Use GPTQ kernels for int8 matrix multiplication
- Write own kernels for int8 matrix multiplication `:)`


## Quantization

There are multiple levels of quantization, a one of the simplest method is to store the model weights using reduce number of bits and to convert them beck to bf16 during inference. Here the activations are computed in bf16 itself. 

Since most of the GPUs support int8 matrix multiplication, we can also quantize the activations to int8 and perform the matrix multiplication in int8 itself. This can reduce the memory footprint and increase the speed of the model.
But this is not straight forward since int8 matrix multiplication may require custom CUDA kernels. Also, the activation quantization mat overflow. Also quantizing the activations require calibrating the quantization parameters using multiple samples. 

FP8 is only supported in H100 GPUs but storing approximations in fp8 can be accurate than vanilla int8 quantization. The recent QLoRA paper explores different data types, 4-bit Float and 4-bit NormalFloat which again are only used for storage and not for computation.

- Intro to weight quantization: https://freedium.cfd/https://medium.com/m/global-identity-2?redirectUrl=https%3A%2F%2Ftowardsdatascience.com%2Fintroduction-to-weight-quantization-2494701b9c0c

- Holy grail: https://timdettmers.com/2023/01/30/which-gpu-for-deep-learning/
- GPT Fast (Read for good quantization implementation) : https://github.com/pytorch-labs/gpt-fast
- Simple notebook: https://colab.research.google.com/drive/1oDfcLRz2AIgsclkXJHj-5wMvbylr4Nxz#scrollTo=iCsoFvwLrgdu

## Types of SoTA quantization methods

- k-bit scaling laws, basically says that 4bit is best, even better than 8bit: https://arxiv.org/pdf/2212.09720.pdf#page=6.11
    - https://www.youtube.com/watch?v=jyOqtw4ry2w
    - https://freedium.cfd/https://medium.com/@metechsolutions/llm-by-examples-use-bitsandbytes-for-quantization-cf33aa8bfe16

- GGUF: mainly bock quantization for use with CPU only: https://kaitchup.substack.com/p/gguf-quantization-for-fast-and-memory
    - GGML format explained: https://freedium.cfd/https://medium.com/m/global-identity-2?redirectUrl=https%3A%2F%2Ftowardsdatascience.com%2Fquantize-llama-models-with-ggml-and-llama-cpp-3612dfbcc172
- AWQ: Activation aware quantization: Uses the distribution of activations to quantize them.
- GPTQ: https://arxiv.org/pdf/2210.17323.pdf
    - Uses 4bit quantization and 16bit computation, the difference with gguf is that it uses a different quantization method.
    - Explanation video: https://www.youtube.com/watch?v=05v2MA3CXKo

- Smooth Quantization+, 4 bit quantization: https://arxiv.org/pdf/2312.03788.pdf
    - https://www.youtube.com/watch?v=RGUCmX1fvOE

- 6bit quantization: https://arxiv.org/pdf/2310.05079.pdf
- QLLM, recent SoTA 4bit: https://arxiv.org/pdf/2310.08041.pdf
- OmniQuant, recent SoTA method: Both weight and activation quantization: https://github.com/OpenGVLab/OmniQuant?tab=readme-ov-file

- Comparison of quantization methods:
    - https://oobabooga.github.io/blog/posts/gptq-awq-exl2-llamacpp/
    - https://freedium.cfd/https://medium.com/m/global-identity-2?redirectUrl=https%3A%2F%2Ftowardsdatascience.com%2Fwhich-quantization-method-is-right-for-you-gptq-vs-gguf-vs-awq-c4cd9d77d5be

- Old quant method: https://github.com/yhhhli/BRECQ

### Ideas

- Start with simple quantization for storage and see the impact on inference latency and quality. (Since model is small, no outliers are there)
- Use bnb to check int8 and int4 quantization.

- INT8 quantization since RTX20 and future GPUs support it. But may require CUDA kernels.
    - llm.int8() paper https://arxiv.org/pdf/2208.07339.pdf
    - Explanation of 8bit: https://huggingface.co/blog/hf-bitsandbytes-integration

- Check the activation distributions
- Check Activation aware quantization. (Mabe too much work)


### General Points

- GPUs support int8 matrix multiplication which is used in llm.int8() by Tim Detmers.
- Only H100 have native fp8 support which is fast but not much use since limited to H100.
- Using torch autocast/amp is simple and good to convert between dtypes, maybe better techniques exist but this is also good


## Other general Optimizations

- https://pytorch.org/blog/accelerating-generative-ai-3/
- https://pytorch.org/blog/accelerating-generative-ai-2/
- Compile with max auto-tune.
- Compute QKV in one go.

### Getting rid of GPU syncs after compilation

During the iterative reverse diffusion process, we call step() on the scheduler each time after the denoiser predicts the less noisy latent embeddings. Inside step(), the sigmas variable is indexed. If the sigmas array is placed on the GPU, indexing causes a communication sync between the CPU and GPU. This causes a latency, and it becomes more evident when the denoiser has already been compiled.

But if the sigmas array always stays on the CPU (refer to this line), this sync doesn’t take place, hence improved latency. In general, any CPU <-> GPU communication sync should be none or be kept to a bare minimum as it can impact inference latency.

## Quantize Diffusion
- https://github.com/Xiuyu-Li/q-diffusion/tree/master
 - https://www.youtube.com/watch?v=virARwF_pt4&t=1669s
- SD3 paper: https://arxiv.org/pdf/2403.03206.pdf

## Read

- Float8 in Pytorch: https://dev-discuss.pytorch.org/t/float8-in-pytorch-1-x/1815
- 4but QLoRA: https://huggingface.co/blog/4bit-transformers-bitsandbytes

## Libraries

- https://github.com/huggingface/quanto

## CUDA references

- https://github.com/IST-DASLab/marlin
- https://github.com/TimDettmers/bitsandbytes
- https://github.com/turboderp/exllama/tree/master/exllama_ext/cuda_func

## Good discussions

- https://github.com/huggingface/quanto/issues/65
- 4/8 bit in diffuser: https://github.com/huggingface/diffusers/issues/6500
- fp8 storage: https://github.com/AUTOMATIC1111/stable-diffusion-webui/pull/14031
- 4bit Qlinear: https://github.com/huggingface/quanto/issues/65
- QX4: https://github.com/ggerganov/llama.cpp/issues/1240
- Quantized linear layer: https://discuss.pytorch.org/t/understanding-quantized-linear-layer/154000
- GPTQ & bnb benchmarking by TheBloke: https://github.com/AutoGPTQ/AutoGPTQ/issues/49#issuecomment-1538065985

## Misc

### FP8 vs INT8
Qualcomm [whitepaper](https://www.qualcomm.com/news/onq/2023/04/floating-point-arithmetic-for-ai-inference-hit-or-miss) shows that the hardware implementation of the FP8 format is somewhere between 50% to 180% less efficient than INT8 in terms of chip area and energy usage. This is because of the additional logic needed in the accumulation of FP formats versus integer formats. This seems like a broad range, but the actual efficiency depends on many hardware design choices that vary greatly. A similar conclusion was reached recently by Microsoft and Meta: Floating-point arithmetic is just much less efficient than integer arithmetic.

This means that FP8 will have to be significantly more accurate than INT8 to be worthwhile from a hardware-efficiency perspective.  

### Quantizing bias

Biases are not converted because to preserve the accuracy of a typical addmm operation, they must be converted with a scale that is equal to the product of the input and weight scales, which leads to a ridiculously small scale, and conversely requires a very high bitwidth to avoid clipping. 