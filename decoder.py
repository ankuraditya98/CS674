import torch as th
import torch.nn.functional as F
from utils.helpers import Net

class VertexUnet(Net):
    def __init__(self, classes: int = 128, heads: int = 64, n_vertices: int = 6172, mean: th.Tensor = None,
                 stddev: th.Tensor = None, model_name = 'vertex_unet'):

        super().__init__(model_name)

        self.classes = classes
        self.heads = heads
        self.n_vertices = n_vertices

        # Encoder Layers
        self.encoder = th.nn.MyModule([
            th.nn.Linear(n_vertices * 3, 512),
            th.nn.Linear(512, 256),
            th.nn.Linear(256, 128)
         ])


        self.fuse = th.nn.Linear(heads*classes +128,128)

        self.lstm = th.nn.LSTM(input_size=128, hidden_size=128, num_layers=2)

        #Decoder Layers
        self.decoder = th.nn.MyModule([
            th.nn.Linear(128, 256),
            th.nn.Linear(256, 512),
            th.nn.Linear(512, n_vertices*3)
         ])
        self.vertex_bias = th.nn.Parameter(th.zeros(n_vertices * 3))

    def forward(self, geom: th.Tensor, expression_encoding: th.Tensor):
        x = (geom - self.mean) / self.stddev
        x = x.view(x.shape[0], x.shape[1], self.n_vertices * 3)

        #Encoding
        skip_e = []
        for i, layer in enumerate(self.encoder):
            skip_e = [x] +skip_e
            geom_encoding = F.leaky_relu(layer(x), 0.2)

        #Fusion
        x = self.fusion(th.cat([geom_encoding, expression_encoding], dim=-1))
        x = F.leaky_relu(x, 0.2)

        #Decoding

        x = self.lstm(x)
        for i, layer in enumerate(self.decoder):
            x = skip_e[i] + F.leaky_relu(layer(x), 0.2)
        x = x + self.vertex_bias.view(1, 1, -1)
        #------ vertex bias reqd? --------

        x = x.view(x.shape[0], x.shape[1], self.n_vertices, 3)
        geom = x * self.stddev + self.mean

        return geom





