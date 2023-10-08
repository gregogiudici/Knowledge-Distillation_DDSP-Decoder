import torch
import torch.nn as nn
import math

def exp_sigmoid(x):
    return 2 * torch.sigmoid(x)**(math.log(10)) + 1e-7

class HpNSynth(nn.Module):
    ''' Harmonic + Noise Synthesizer (with Reverb) '''
    def __init__(self, sample_rate, block_size, reverb_scale = 1, scale_fn = exp_sigmoid):
        super().__init__()
        self.sample_rate = sample_rate
        self.block_size = block_size
        self.reverb = Reverb(length=int(sample_rate/reverb_scale), sample_rate=sample_rate)
        self.scale_fn = scale_fn

    # expects: harmonic_distr, amplitude, noise_bands
    def forward(self,controls):

        harmonics = self.scale_fn(controls['harmonic_distribution'])
        noise_bands = self.scale_fn(controls['noise_bands'])
        total_amp = self.scale_fn(controls['amplitude'])

        harmonics = remove_above_nyquist(
            harmonics,
            controls['f0_hz'],
            self.sample_rate,
        )
        harmonics /= harmonics.sum(-1, keepdim=True)
        harmonics *= total_amp

        harmonics_up = upsample(harmonics, self.block_size)
        f0_up = upsample(controls['f0_hz'], self.block_size,'linear')

        harmonic_signal = harmonic_synth(f0_up, harmonics_up, self.sample_rate)
        impulse = amp_to_impulse_response(noise_bands, self.block_size)

        noise = torch.rand(
            impulse.shape[0],
            impulse.shape[1],
            self.block_size,
            ).to(impulse) * 2 - 1

        noise = fft_convolve(noise, impulse).contiguous()
        noise = noise.reshape(noise.shape[0], -1, 1)

        dereverb_signal = harmonic_signal + noise

        # Reverb part
        synth_signal = self.reverb(dereverb_signal)
        synth_out = {
            'synth_audio': synth_signal,
            'dereverb_audio' : dereverb_signal,
            'noise_audio' : noise,
            'harmonic_audio' : harmonic_signal,
            'harmonic_distribution': harmonics,
            'noise_bands': noise_bands,
            'f0_hz': controls['f0_hz']
            }

        return synth_out
    


class Reverb(nn.Module):
    ''' Reverb '''
    def __init__(self, length, sample_rate, initial_wet=0, initial_decay=5):
        super().__init__()
        self.length = length
        self.sample_rate = sample_rate

        self.noise = nn.Parameter((torch.rand(length) * 2 - 1).unsqueeze(-1))
        self.decay = nn.Parameter(torch.tensor(float(initial_decay)))
        self.wet = nn.Parameter(torch.tensor(float(initial_wet)))

        t = torch.arange(self.length) / self.sample_rate
        t = t.reshape(1, -1, 1)
        self.register_buffer("t", t)

    def build_impulse(self):
        t = torch.exp(-nn.functional.softplus(-self.decay) * self.t * 500)
        noise = self.noise * t
        impulse = noise * torch.sigmoid(self.wet)
        impulse[:, 0] = 1
        return impulse

    def forward(self, x):
        lenx = x.shape[1]
        impulse = self.build_impulse()
        impulse = nn.functional.pad(impulse, (0, 0, 0, lenx - self.length))

        x = fft_convolve(x.squeeze(-1), impulse.squeeze(-1)).unsqueeze(-1)

        return x


def upsample(signal, factor,mode='nearest'):
    signal = signal.permute(0, 2, 1)
    signal = nn.functional.interpolate(signal, size=signal.shape[-1] * factor,mode=mode)
    return signal.permute(0, 2, 1)

def amp_to_impulse_response(amp, target_size):
    amp = torch.stack([amp, torch.zeros_like(amp)], -1)
    amp = torch.view_as_complex(amp)
    amp = torch.fft.irfft(amp)

    filter_size = amp.shape[-1]

    amp = torch.roll(amp, filter_size // 2, -1)
    win = torch.hann_window(filter_size, dtype=amp.dtype, device=amp.device)

    amp = amp * win

    amp = nn.functional.pad(amp, (0, int(target_size) - int(filter_size)))
    amp = torch.roll(amp, -filter_size // 2, -1)

    return amp


def fft_convolve(signal, kernel):
    signal = nn.functional.pad(signal, (0, signal.shape[-1]))
    kernel = nn.functional.pad(kernel, (kernel.shape[-1], 0))

    output = torch.fft.irfft(torch.fft.rfft(signal) * torch.fft.rfft(kernel))
    output = output[..., output.shape[-1] // 2:]

    return output


def harmonic_synth(pitch, amplitudes, sampling_rate):

    omega = torch.cumsum(2 * torch.pi * pitch / sampling_rate, 1)

    n_harmonic = amplitudes.shape[-1]
    omegas = omega * torch.arange(1, n_harmonic + 1).to(omega)
    signal = (torch.sin(omegas) * amplitudes).sum(-1, keepdim=True)
    return signal


def remove_above_nyquist(amplitudes, pitch, sampling_rate):
    n_harm = amplitudes.shape[-1]
    pitches = pitch * torch.arange(1, n_harm + 1).to(pitch)
    aa = (pitches < sampling_rate / 2).float() + 1e-4
    return amplitudes * aa

