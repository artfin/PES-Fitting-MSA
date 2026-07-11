"""
Shared configuration for PES-Fitting-MSA.
"""

import numpy as np
import torch

# Floating point precision: "float32" (reduced memory) or "float64" (full precision)
PRECISION = "float32"

NP_FLOAT = np.float32 if PRECISION == "float32" else np.float64
TORCH_FLOAT = torch.float32 if PRECISION == "float32" else torch.float64
C_FLOAT_TYPE = "float" if PRECISION == "float32" else "double"

# Compute device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Logging / reporting
PRINT_TRAINING_STEPS = 1  # log every N epochs
PRINT_PRECISION = 3       # decimal places in reported losses

# Weights & Biases integration (disabled by default)
USE_WANDB = False
