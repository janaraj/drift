"""Cloud open-weight target served via vLLM (CUDA).

Used for training rollouts at scale on a cloud GPU box. Same Target protocol
as local_mlx.py; the dialogue env does not care which backend serves.

Filled in when a cloud box is provisioned (Unit 1.1 cloud half).
"""
