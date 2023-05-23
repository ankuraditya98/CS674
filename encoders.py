import torch.nn as nn


class ExpressionEncoder(nn.Module):
    """
    ExpressionEncoder to extract latent spaced from 3D mesh data
    """
    def __init__(self):
        super(ExpressionEncoder, self, latent_dim = 128, n_vertices = 6172, model_name = 'expression_encoder').__init__()

        self.fc1 = nn.Linear(n_vertices * 3, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, latent_dim)
        self.lstm = nn.LSTM(128, 128, batch_first=True)
        
    def forward(self, x):
        
        x = nn.functional.normalize(x)
        x = x.reshape(x.shape[0], x.shape[1] * 3) # flatten

        x = nn.functional.leaky_relu(self.fc1(x), 0.2)
        x = nn.functional.leaky_relu(self.fc2(x), 0.2)

        x, _ = self.lstm(x)
        x = self.fc3(x)

        return x
        
