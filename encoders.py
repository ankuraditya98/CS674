import torch.nn as nn


class Expression_Encoder(nn.Module):
    """
    A PixelCNN variant you have to implement according to the instructions
    """

    def __init__(self):
        super(PixelCNN, self).__init__()

        # WRITE CODE HERE TO IMPLEMENT THE MODEL STRUCTURE
        
        self.maskedconv1 = MaskedCNN(mask_type='A', in_channels=1, out_channels=16, kernel_size=3, dilation=3, padding = 3, padding_mode='reflect', bias=False)
        self.batchnorm1 = nn.BatchNorm2d(16)

        self.maskedconv2 = MaskedCNN(mask_type='B', in_channels=16, out_channels=16, kernel_size=3, dilation=3, padding = 3, padding_mode='reflect', bias=False)
        self.batchnorm2 = nn.BatchNorm2d(16)

        self.maskedconv3 = MaskedCNN(mask_type='B', in_channels=16, out_channels=16, kernel_size=3, dilation=3, padding = 3, padding_mode='reflect', bias=False)
        self.batchnorm3 = nn.BatchNorm2d(16)

        self.conv = nn.Conv2d(in_channels=16, out_channels=1, kernel_size=1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        # WRITE CODE HERE TO IMPLEMENT THE FORWARD PASS

        x = self.batchnorm1(self.maskedconv1(x))
        x = nn.functional.leaky_relu(x)

        x = self.batchnorm2(self.maskedconv2(x))
        x = nn.functional.leaky_relu(x)

        x = self.batchnorm3(self.maskedconv3(x))
        x = nn.functional.leaky_relu(x)

        return self.sigmoid(self.conv(x))
        # return self.conv(self.sigmoid(x))
