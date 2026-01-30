import torch.nn as nn

class MLP(nn.Module):
    """Multi-layer perceptron decoder.

    Simple feedforward neural network that maps from latent features
    to output predictions.

    Args:
        in_dim: Input feature dimension
        out_dim: Output dimension (typically 1 for grayscale intensity)
        n_hidden: Number of hidden layers
        n_neurons: Number of neurons per hidden layer
    """
    def __init__(self, in_dim, out_dim, n_hidden, n_neurons):
        super().__init__()

        layers = []
        self.in_dim = in_dim
        
        self.out_dim = out_dim
        lastv = self.in_dim

        for i in range(n_hidden):
            layers.append(nn.Linear(lastv, n_neurons))
            layers.append(nn.ReLU())
            lastv = n_neurons
        layers.append(nn.Linear(lastv, out_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        """Forward pass through MLP.

        Args:
            x: Input tensor of any shape ending with in_dim

        Returns:
            Output tensor with same shape except last dim is out_dim
        """
        shape = x.shape[:-1]
        x = self.layers(x.view(-1, x.shape[-1]))
        return x.view(*shape, -1)
