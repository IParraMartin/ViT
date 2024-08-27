# Initialize weights
import torch.nn as nn
import math

# BERT-like weight initialization
def init_weights(model: nn.Module) -> None:

    flag_linear = False
    flag_convolutional = False
    flag_multi_head_attn = False
    flag_layernorm = False
    flag_cls_token = False
    flag_pos_embeddings = False

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
            if m.bias is not None:
                nn.init.zeros_(m.bias)
            if not flag_convolutional:
                print("Convolutional weights initialized")
                flag_convolutional = True

        elif isinstance(m, nn.MultiheadAttention):
            nn.init.normal_(m.in_proj_weight, mean=0.0, std=0.02) # We can also try to initialize as they do in the GPT2 paper (std=0.02/math.sqrt(2 * n_layers))
            if m.in_proj_bias is not None:
                nn.init.zeros_(m.in_proj_bias)
            if not flag_multi_head_attn:
                print("MHA weights initialized")
                flag_multi_head_attn = True

        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
            if not flag_layernorm:
                print("LayerNorm weights initialized")
                flag_layernorm = True

    # Initializing the CLS and Positional Embeddings is less trivial:
    for name, param in model.named_parameters():

        if "cls_token" in name and param.shape == (1, 1, model.d_model):
            nn.init.normal_(param, mean=0, std=0.02)
            if not flag_cls_token:
                print("CLS Token weights initialized")
                flag_cls_token = True
        
        elif "pos_embedding" in name and param.shape == (1, 1 + model.patch_embed.n_patches, model.d_model):
            nn.init.normal_(param, mean=0, std=0.02)
            if not flag_pos_embeddings:
                print("Positional Embedding weights initialized")
                flag_pos_embeddings = True
