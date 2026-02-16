import torch
import torch.nn as nn
import torch.nn.functional as F
import math
class LoRALinear(nn.Module):
        def __init__(self, original_linear, r=8, alpha=16, dropout=0.1):
            super().__init__()
            self.original_linear = original_linear
            self.r = r
            self.alpha = alpha
            self.dropout = nn.Dropout(dropout)

            self.lora_A = nn.Linear(original_linear.in_features, r, bias=False)
            self.lora_B = nn.Linear(r, original_linear.out_features, bias=False)

            # Scale factor
            self.scaling = self.alpha / self.r

            # Init
            nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B.weight)

        def forward(self, x):
            return self.original_linear(x) + self.dropout(self.lora_B(self.lora_A(x))) * self.scaling
    # ---------------------------