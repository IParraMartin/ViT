# Initialize weights
import torch.nn as nn

# BERT-like weight initialization
def init_weights(model: nn.Module) -> None:

    flag_linear = False
    flag_convolutional = False
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

    # Handle nn.Parameter for cls_token and pos_embedding separately
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

