import torch
import torch.nn as nn


class BaseCNN(nn.Module):
    def __init__(self, in_channels, out_channels=32, n_filters=32, groups=1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_filters = n_filters
        self.groups = groups
        self.cnn = None
        self.build_model()

    def build_model(self):
        raise NotImplementedError

    def forward(self, x):
        return self.cnn(x)

    def infer_output_size(self, input_size):
        sample_input = torch.zeros(1, self.in_channels, input_size, input_size)
        with torch.no_grad():
            output = self.cnn(sample_input)

        return output.shape[-1]


class NatureEncoder(BaseCNN):
    def build_model(self):
        self.cnn = nn.Sequential(
            nn.Conv2d(self.in_channels, self.n_filters, kernel_size=8, stride=4, groups=self.groups),
            nn.ReLU(),
            nn.Conv2d(self.n_filters, self.n_filters * 2, kernel_size=4, stride=2, groups=self.groups),
            nn.ReLU(),
            nn.Conv2d(self.n_filters * 2, self.out_channels, kernel_size=3, stride=1, groups=self.groups)
        )


class NatureDecoder(BaseCNN):
    def build_model(self):
        self.cnn = nn.Sequential(
            nn.ConvTranspose2d(self.in_channels, self.n_filters * 2, kernel_size=3, stride=1, groups=self.groups),
            nn.ReLU(),
            nn.ConvTranspose2d(self.n_filters * 2, self.n_filters, kernel_size=4, stride=2, groups=self.groups),
            nn.ReLU(),
            nn.ConvTranspose2d(self.n_filters, self.out_channels, kernel_size=8, stride=4, groups=self.groups)
        )


class CustomeEncoder(BaseCNN):
    def build_model(self):
        self.cnn = nn.Sequential(
            nn.Conv2d(self.in_channels, self.n_filters, kernel_size=4, stride=2, groups=self.groups),
            nn.ReLU(),
            nn.Conv2d(self.n_filters, self.n_filters * 2, kernel_size=3, stride=2, groups=self.groups),
            nn.ReLU(),
            nn.Conv2d(self.n_filters * 2, self.out_channels, kernel_size=4, stride=2, groups=self.groups),
        )


class CustomeDecoder(BaseCNN):
    def build_model(self):
        self.cnn = nn.Sequential(
            nn.Conv2d(self.in_channels, self.n_filters * 4, kernel_size=3, stride=1, padding=1, groups=self.groups),
            nn.ReLU(),
            nn.ConvTranspose2d(self.n_filters * 4, self.n_filters * 2, kernel_size=4, stride=2, groups=self.groups),
            nn.ReLU(),
            nn.ConvTranspose2d(self.n_filters * 2, self.n_filters, kernel_size=3, stride=2, groups=self.groups),
            nn.ReLU(),
            nn.ConvTranspose2d(self.n_filters, self.out_channels, kernel_size=4, stride=2, groups=self.groups)
        )


cnn_registry = dict(
    nature=[NatureEncoder, NatureDecoder],
    custome=[CustomeEncoder, CustomeDecoder]
)


if __name__ == '__main__':
    encoder = CustomeEncoder(1)
    decoder = CustomeDecoder(32)
    latent = encoder(torch.zeros(1, 1, 108, 128))
    print(latent.shape)
    output = decoder(torch.zeros(1, 32, 12, 14))
    print(latent.shape, output.shape)
