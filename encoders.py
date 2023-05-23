import torch.nn as nn
import torch.cat as cat


class ExpressionEncoder(nn.Module):
    """
    ExpressionEncoder to extract a latent space from 3D mesh data
    """
    def __init__(self):
        super(ExpressionEncoder, self, latent_dim = 128, n_vertices = 6172).__init__()

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
        
class FusionEncoder():
    """
    FusionEncoder to extract a combined latent space from both of audio and expression latent spaces
    """
    def __init__(self):
        super(ExpressionEncoder, self, latent_dim = 2*128, heads = 64, categories = 128).__init__()
        
        # using a simple 3 layers MLP block to combine latent spaces
        self.mlp = nn.Sequential(
                nn.Linear(latent_dim, latent_dim),
                nn.LeakyReLU(0.2),
                nn.Linear(latent_dim, latent_dim)
                nn.LeakyReLU(0.2),
                nn.Linear(latent_dim, output_size),
            )

    def forward(self, audioSpace, expressionSpace):
        
        # concat latent spaces
        x = cat((expressionSpace, audioSpace), -1)
        x = self.mlp(x).reshape(x.shape[0], x.shape[1], self.head, self.categories)
        x = nn.functional.gumbel_softmax(x, tau=2, hard=False, dim=-1) #TODO: test tau=1?

        return x