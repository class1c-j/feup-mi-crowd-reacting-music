import torch
from pypianoroll import Multitrack, BinaryTrack
import numpy as np
import matplotlib.pyplot as plt

latent_dim = 128  # Define the dimension of the latent vector
latent = torch.randn(1, latent_dim)  # Generate a random latent vector
beat_resolution = 4
lowest_pitch = 24
n_pitches = 72
n_tracks = 5
n_measures = 4
measure_resolution = 4 * beat_resolution
n_samples_per_song = 8
tempo = 100
tempo_array = np.full((4 * 4 * measure_resolution, 1), tempo)
n_samples = 4
sample_latent = torch.randn(n_samples, latent_dim)
programs = [0, 0, 25, 33, 48]
is_drums = [True, False, False, False, False]
track_names = ["Drums", "Piano", "Guitar", "Bass", "Strings"]
tempo = 100
tempo_array = np.full((4 * 4 * measure_resolution, 1), tempo)


class GeneratorBlock(torch.nn.Module):
    def __init__(self, in_dim, out_dim, kernel, stride):
        super().__init__()
        self.transconv = torch.nn.ConvTranspose3d(in_dim, out_dim, kernel, stride)
        self.batchnorm = torch.nn.BatchNorm3d(out_dim)

    def forward(self, x):
        x = self.transconv(x)
        x = self.batchnorm(x)
        return torch.nn.functional.relu(x)


class Generator(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.transconv0 = GeneratorBlock(latent_dim, 256, (4, 1, 1), (4, 1, 1))
        self.transconv1 = GeneratorBlock(256, 128, (1, 4, 1), (1, 4, 1))
        self.transconv2 = GeneratorBlock(128, 64, (1, 1, 4), (1, 1, 4))
        self.transconv3 = GeneratorBlock(64, 32, (1, 1, 3), (1, 1, 1))
        self.transconv4 = torch.nn.ModuleList(
            [GeneratorBlock(32, 16, (1, 4, 1), (1, 4, 1)) for _ in range(n_tracks)]
        )
        self.transconv5 = torch.nn.ModuleList(
            [GeneratorBlock(16, 1, (1, 1, 12), (1, 1, 12)) for _ in range(n_tracks)]
        )

    def forward(self, x):
        x = x.view(-1, latent_dim, 1, 1, 1)
        x = self.transconv0(x)
        x = self.transconv1(x)
        x = self.transconv2(x)
        x = self.transconv3(x)
        x = [transconv(x) for transconv in self.transconv4]
        x = torch.cat([transconv(x_) for x_, transconv in zip(x, self.transconv5)], 1)
        x = x.view(-1, n_tracks, n_measures * measure_resolution, n_pitches)
        return x


class DiscriminatorBlock(torch.nn.Module):
    def __init__(self, in_dim, out_dim, kernel, stride):
        super().__init__()
        self.transconv = torch.nn.Conv3d(in_dim, out_dim, kernel, stride)
        self.layernorm = LayerNorm(out_dim)

    def forward(self, x):
        x = self.transconv(x)
        x = self.layernorm(x)
        return torch.nn.functional.leaky_relu(x)


class Discriminator(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv0 = torch.nn.ModuleList(
            [DiscriminatorBlock(1, 16, (1, 1, 12), (1, 1, 12)) for _ in range(n_tracks)]
        )
        self.conv1 = torch.nn.ModuleList(
            [DiscriminatorBlock(16, 16, (1, 4, 1), (1, 4, 1)) for _ in range(n_tracks)]
        )
        self.conv2 = DiscriminatorBlock(16 * n_tracks, 64, (1, 1, 3), (1, 1, 1))
        self.conv3 = DiscriminatorBlock(64, 64, (1, 1, 4), (1, 1, 4))
        self.conv4 = DiscriminatorBlock(64, 128, (1, 4, 1), (1, 4, 1))
        self.conv5 = DiscriminatorBlock(128, 128, (2, 1, 1), (1, 1, 1))
        self.conv6 = DiscriminatorBlock(128, 256, (3, 1, 1), (3, 1, 1))
        self.dense = torch.nn.Linear(256, 1)

    def forward(self, x):
        x = x.view(-1, n_tracks, n_measures, measure_resolution, n_pitches)
        x = [conv(x[:, [i]]) for i, conv in enumerate(self.conv0)]
        x = torch.cat([conv(x_) for x_, conv in zip(x, self.conv1)], 1)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = x.view(-1, 256)
        x = self.dense(x)
        return x


class LayerNorm(torch.nn.Module):
    """An implementation of Layer normalization that does not require size
    information. Copied from https://github.com/pytorch/pytorch/issues/1959."""

    def __init__(self, n_features, eps=1e-5, affine=True):
        super().__init__()
        self.n_features = n_features
        self.affine = affine
        self.eps = eps
        if self.affine:
            self.gamma = torch.nn.Parameter(torch.Tensor(n_features).uniform_())
            self.beta = torch.nn.Parameter(torch.zeros(n_features))

    def forward(self, x):
        shape = [-1] + [1] * (x.dim() - 1)
        mean = x.view(x.size(0), -1).mean(1).view(*shape)
        std = x.view(x.size(0), -1).std(1).view(*shape)
        y = (x - mean) / (std + self.eps)
        if self.affine:
            shape = [1, -1] + [1] * (x.dim() - 2)
            y = self.gamma.view(*shape) * y + self.beta.view(*shape)
        return y


def generate_music(quadrant, display=False):
    generator = Generator()
    model_name = "./music_models/q" + str(quadrant) + "_model.pt"
    checkpoint = torch.load(model_name, map_location=torch.device("cpu"))
    generator.load_state_dict(checkpoint["generator_state_dict"])
    generator.eval()  # Set the generator to evaluation mode

    samples = generator(sample_latent).detach().cpu().numpy()
    samples = samples.transpose(1, 0, 2, 3).reshape(n_tracks, -1, n_pitches)
    tracks = []
    for idx, (program, is_drum, track_name) in enumerate(
        zip(programs, is_drums, track_names)
    ):
        pianoroll = np.pad(
            samples[idx] > 0.5, ((0, 0), (lowest_pitch, 128 - lowest_pitch - n_pitches))
        )
        tracks.append(
            BinaryTrack(
                name=track_name, program=program, is_drum=is_drum, pianoroll=pianoroll
            )
        )
    m = Multitrack(tracks=tracks, tempo=tempo_array, resolution=beat_resolution)

    if display:
        axs = m.plot()
        plt.gcf().set_size_inches((16, 8))
        for ax in axs:
            for x in range(
                measure_resolution,
                4 * measure_resolution * n_measures,
                measure_resolution,
            ):
                if x % (measure_resolution * 4) == 0:
                    ax.axvline(x - 0.5, color="k")
                else:
                    ax.axvline(x - 0.5, color="k", linestyle="-", linewidth=1)
        plt.show()

    # Convert Multitrack to PrettyMIDI
    midi_music = m.to_pretty_midi()
    midi_music.write("current.mid")

