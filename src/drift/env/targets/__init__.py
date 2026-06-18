"""Target adapters — local (MLX), cloud (vLLM), and API (Anthropic, OpenAI, Google).

Imports here register each adapter in the target registry. Adding a new
target means dropping a new module beside the existing ones and importing it.
"""

# Imports for registration side effects. All adapters use lazy SDK imports, so
# importing them here does not require any backend/SDK to be installed.
from drift.env.targets import (  # noqa: F401
    api_anthropic,
    api_google,
    api_openai,
    local_mlx,
)
