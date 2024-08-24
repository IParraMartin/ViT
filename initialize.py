# Initialize weights
import torch.nn as nn

# BERT-like weight initialization
def init_weights(model: nn.Module) -> None:
    flag_linear = False
    flag_embedding = False
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
            if not flag_linear:
                print("Linear weights initialized")
                flag_linear = True
        elif isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if not flag_embedding:
                print("Embedding weights initialized")
                flag_embedding = True