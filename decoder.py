import torch
from torch import nn
import torch.nn.functional as F
from helpers import Net

class VertexUNet(Net):
    def __init__(self, classes: int = 128, heads: int = 64, n_vertices: int = 6172, mean: torch.Tensor = None,
                 stddev: torch.Tensor = None, model_name = 'vertex_unet'):

        super().__init__(model_name)

        self.classes = classes
        self.heads = heads
        self.n_vertices = n_vertices

        # Encoder Layers
        self.encoder = nn.ModuleList([
            nn.Linear(n_vertices * 3, 512),
            nn.Linear(512, 256),
            nn.Linear(256, 128)
         ])


        self.fuse = nn.Linear(heads*classes +128,128)

        self.lstm = nn.LSTM(input_size=128, hidden_size=128, num_layers=2)

        #Decoder Layers
        self.decoder = nn.ModuleList([
            nn.Linear(128, 256),
            nn.Linear(256, 512),
            nn.Linear(512, n_vertices*3)
         ])
        self.vertex_bias = nn.Parameter(torch.zeros(n_vertices * 3))

    def forward(self, geom: torch.Tensor, expression_encoding: torch.Tensor):
        x = (geom - self.mean) / self.stddev
        x = x.view(x.shape[0], x.shape[1], self.n_vertices * 3)

        #Encoding
        skip_e = []
        for i, layer in enumerate(self.encoder):
            skip_e = [x] +skip_e
            geom_encoding = F.leaky_relu(layer(x), 0.2)

        #Fusion
        x = self.fusion(torch.cat([geom_encoding, expression_encoding], dim=-1))
        x = F.leaky_relu(x, 0.2)

        #Decoding

        x, _ = self.lstm(x)
        for i, layer in enumerate(self.decoder):
            x = skip_e[i] + F.leaky_relu(layer(x), 0.2)
        x = x + self.vertex_bias.view(1, 1, -1)
        #------ vertex bias reqd? --------

        x = x.view(x.shape[0], x.shape[1], self.n_vertices, 3)
        geom = x * self.stddev + self.mean

        return geom





