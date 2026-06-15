"""Local open-weight target served via MLX / mlx-lm (Apple Silicon dev).

vLLM does not run on Metal, so local serving on the M4 Pro dev box uses MLX.
The CUDA cloud equivalent lives in cloud_vllm.py. Both implement the same
Target protocol — the dialogue env is agnostic to which backend serves.

Filled in by Unit 1.1.
"""
