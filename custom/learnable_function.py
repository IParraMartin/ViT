import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


"""
Introduce a learnable parameter to the Swish activation function.
TO DO:
    - Try with other base functions
"""

class LearnableNonLinearity(nn.Module):

    def __init__(self, base: torch.nn.functional = F.sigmoid, alpha: float = 1.0):
        super().__init__()
        self.base = base
        self.alpha = nn.Parameter(torch.tensor(alpha))

    def forward(self, x):
        return x * self.base(self.alpha * x)


if __name__ == '__main__':

    x = torch.linspace(-10, 10, 100)
    plt.figure(figsize=(10, 6))
    alphas = [0.1, 0.5, 1.0, 2.0, 5.0]
    colors = ['b', 'g', 'r', 'c', 'm']
    for i, alpha in enumerate(alphas):
        learnable_swish = LearnableNonLinearity(base=F.gelu, alpha=alpha)
        y = learnable_swish(x).detach().numpy()
        plt.plot(x.detach().numpy(), y, label=f'alpha={alpha}')
    plt.legend()
    plt.grid()
    plt.title('Learnable Swish Activation Function')
    plt.show()
