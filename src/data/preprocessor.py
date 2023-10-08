
import torch

_DB_RANGE = 80.0 # Min loudness
_F0_RANGE = 127

class F0LoudnessRMSPreprocessor():
    """Scales 'f0_hz' and 'loudness_db' features."""
    def __init__(self):
        return

    def run(self,x):
        x['loudness_scaled'] = self.scale_db(x['loudness'])
        x['rms_scaled'] = self.scale_db(x['rms'])
        x['f0_scaled'] = self.scale_f0_hz(x['f0'])
        return x

    def scale_db(self,db):
        """Scales [-DB_RANGE, 0] to [0, 1]."""
        return (db / _DB_RANGE) + 1.0

    def scale_f0_hz(self,f0_hz):
        """Scales [0, Nyquist] Hz to [0, 1.0] MIDI-scaled."""
        return hz_to_midi(f0_hz) / _F0_RANGE


def hz_to_midi(frequencies):
    """Torch-compatible hz_to_midi function."""
    notes = 12.0 * (logb(frequencies, 2.0) - logb(440.0, 2.0)) + 69.0
    notes = torch.where(torch.le(frequencies, torch.zeros(1).to(frequencies)),
                        torch.zeros(1).to(frequencies), notes)
    return notes


def safe_log(x,eps=1e-7):
    eps = torch.tensor(eps)
    return torch.log(x + eps)

def safe_divide(numerator, denominator, eps=1e-7):
    """Avoid dividing by zero by adding a small epsilon."""
    eps = torch.tensor(eps)
    safe_denominator = torch.where(denominator == 0.0, eps, denominator)
    return numerator / safe_denominator

def logb(x, base=2.0, eps=1e-5):
    """Logarithm with base as an argument."""
    return safe_divide(safe_log(x, eps), safe_log(base, eps), eps)

