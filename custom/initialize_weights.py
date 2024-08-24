import torch.nn as nn


def initialize_weights(model: nn.Module) -> None:

    conv2d_init_printed = False
    batchnorm2d_init_printed = False
    linear_init_printed = False

    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            if model.activation == 'leaky_relu':
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if not conv2d_init_printed:
                    print("Conv2D layers using 'Kaiming Normal' weight initialization")
                    conv2d_init_printed = True
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            # GELU is an approximation of the ReLU activation function, so we can use the same initialization
            elif model.activation == 'relu' or model.activation == 'gelu' or model.activation == 'learnable':
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if not conv2d_init_printed:
                    print("Conv2D using 'Kaiming Normal' weight initialization")
                    conv2d_init_printed = True
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        if isinstance(m, nn.BatchNorm2d):
            if not batchnorm2d_init_printed:
                print("BatchNorm2d layers using 'Kaiming Normal' weight initialization")
                batchnorm2d_init_printed = True
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        if isinstance(m, nn.Linear):
            if not linear_init_printed:
                print("Linear layers using 'Kaiming Normal' weight initialization")
                linear_init_printed = True
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)