import torch.nn as nn
import torch
import torch.nn.functional as F
import torchaudio as ta

from utils.helpers import Net


class AudioEncoder(Net):
    def __init__(self, l_dim: int = 128, model_name: str = 'audio_encoder'):
        super().__init__(model_name)

        self.melspec = ta.transforms.MelSpectrogram(sample_rate=16000, n_fft=2048, win_length=800, hop_length=160, n_mels=80)
        self.cd = torch.nn.Conv1d(in_channels = 80, out_channels = 128, kernel_size = 5)
        torch.nn.init.xavier_uniform_(self.cd)

        self.conv1 = torch.nn.Conv1d(in_channels=128, out_channels=128, kernel_size=5, dilation=2)
        torch.nn.init.xavier_uniform_(self.conv1)
        self.conv2 = torch.nn.Conv1d(in_channels = 128, out_channels=128, kernel_size=5, dilation=4)
        torch.nn.init.xavier_uniform_(self.conv2)
        self.conv3 = torch.nn.Conv1d(in_channels=128, out_channels=128, kernel_size=5, dilation=6)
        torch.nn.init.xavier_uniform_(self.conv3)
        self.conv4 = torch.nn.Conv1d(in_channels=128, out_channels=128, kernel_size=5, dilation=2)
        torch.nn.init.xavier_uniform_(self.conv4)
        self.conv5 = torch.nn.Conv1d(in_channels=128, out_channels=128, kernel_size=5, dilation=4)
        torch.nn.init.xavier_uniform_(self.conv5)
        self.conv6 = torch.nn.Conv1d(in_channels=128, out_channels=128, kernel_size=5, dilation=6)
        torch.nn.init.xavier_uniform_(self.conv6)

        self.fc = torch.nn.Linear(in_features=128, out_features=128)
        torch.nn.init.xavier_uniform_(self.fc)

    def forward(self, audio: torch.Tensor):
        B, T = audio.shape[0], audio.shape[1]

        x = self.melspec(audio).squeeze(1)
        x = torch.log(x.clamp(min=1e-10, max=None))
        if T == 1:
            x = x.unsqueeze(1)

        # Convert to the right dimensionality
        x = x.view(-1, x.shape[2], x.shape[3])
        x = F.leaky_relu(self.cd(x), .2)

        # Forward Pass of Layers
        x1 = F.leaky_relu(self.conv1(x), negative_slope=0.2)
        x1 = F.dropout(x1, 0.2)
        pad = (x.shape[2] - x1.shape[2]) // 2
        x = (x[:, :, pad:-pad] + x1) / 2

        x2 = F.leaky_relu(self.conv2(x), negative_slope=0.2)
        x2 = F.dropout(x2, 0.2)
        pad = (x.shape[2] - x2.shape[2]) // 2
        x = (x[:, :, pad:-pad] + x2) / 2

        x3 = F.leaky_relu(self.conv3(x), negative_slope=0.2)
        x3 = F.dropout(x3, 0.2)
        pad = (x.shape[2] - x3.shape[2]) // 2
        x = (x[:, :, pad:-pad] + x3) / 2

        x4 = F.leaky_relu(self.conv4(x), negative_slope=0.2)
        x4 = F.dropout(x4, 0.2)
        pad = (x.shape[2] - x4.shape[2]) // 2
        x = (x[:, :, pad:-pad] + x4) / 2

        x5 = F.leaky_relu(self.conv5(x), negative_slope=0.2)
        x5 = F.dropout(x5, 0.2)
        pad = (x.shape[2] - x5.shape[2]) // 2
        x = (x[:, :, pad:-pad] + x5) / 2

        x6 = F.leaky_relu(self.conv6(x), negative_slope=0.2)
        x6 = F.dropout(x6, 0.2)
        pad = (x.shape[2] - x6.shape[2]) // 2
        x = (x[:, :, pad:-pad] + x6) / 2

        x = torch.mean(x, dim=-1)
        x = x.view(B, T, x.shape[-1])
        x = self.code(x)

        return x


class ExpressionEncoder(nn.Module):
    """
    ExpressionEncoder to extract a latent space from 3D mesh data
    """
    def __init__(self, latent_dim = 128, n_vertices = 6172):
        super(ExpressionEncoder, self).__init__()

        self.fc1 = nn.Linear(n_vertices * 3, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, latent_dim)
        self.lstm = nn.LSTM(128, 128, batch_first=True)
        
    def forward(self, x):
        
        x = nn.functional.normalize(x)
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * 3) # flattening from BxTxVx3 to BxTx(V*3)

        x = F.leaky_relu(self.fc1(x), 0.2)
        x = F.leaky_relu(self.fc2(x), 0.2)

        x, _ = self.lstm(x)
        x = self.fc3(x)

        return x
        
class FusionEncoder():
    """
    FusionEncoder to extract a combined latent space from both of audio and expression latent spaces
    """
    def __init__(self, latent_dim = 2*128, heads = 64, categories = 128):
        super(ExpressionEncoder, self).__init__()
        
        # using a simple 3 layers MLP block to combine latent spaces
        self.mlp = nn.Sequential(
                nn.Linear(latent_dim, latent_dim),
                nn.LeakyReLU(0.2),
                nn.Linear(latent_dim, latent_dim),
                nn.LeakyReLU(0.2),
                nn.Linear(latent_dim, output_size),
            )

    def forward(self, audioSpace, expressionSpace):
        
        # concat latent spaces
        x = torch.cat((expressionSpace, audioSpace), -1)
        x = self.mlp(x).reshape(x.shape[0], x.shape[1], self.head, self.categories)
        x = F.gumbel_softmax(x, tau=2, hard=False, dim=-1) #TODO: test tau=1?

        return x