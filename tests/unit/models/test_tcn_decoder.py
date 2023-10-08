import pytest
import torch
from src.models import TCNdecoder


@pytest.fixture
def tcn_decoder():
    return TCNdecoder(
                    input_keys=['f0_scaled','loudness_scaled'],
                    input_sizes=[1,1],
                    output_keys=['out_a','out_b','out_c'],
                    output_sizes=[1,10,100])

def test_tcn_decoder_can_forward(tcn_decoder):
    batch_size = 16
    seq_len = 1000
    decoder_in = {'f0': torch.rand([batch_size,seq_len,1]),
                  'f0_scaled': torch.rand([batch_size,seq_len,1]),
                  'loudness_scaled': torch.rand([batch_size,seq_len,1])}
    output = tcn_decoder(decoder_in)
    assert output['out_a'].shape == (16, 1000, 1)
    assert output['out_b'].shape == (16, 1000, 10)
    assert output['out_c'].shape == (16, 1000, 100)
    assert output['out_a'].requires_grad
    assert output['out_b'].requires_grad
    assert output['out_c'].requires_grad