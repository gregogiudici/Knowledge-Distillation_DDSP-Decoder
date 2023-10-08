from pathlib import Path

import pytest
import torch

from src.data.lightning_datamodule import DecoderDataModule


@pytest.mark.parametrize("batch_size", [8, 18])
def test_URMP_datamodule(batch_size):
    data_dir = "dataset/data"
    instrument = "flute"
    
    dm = DecoderDataModule(data_dir=data_dir, 
                           instrument=instrument,
                           batch_size=batch_size)
    dm.prepare_data()

    assert not dm.data_train and not dm.data_val and not dm.data_test

    dm.setup()
    assert dm.data_train and dm.data_val and dm.data_test
    assert dm.train_dataloader() and dm.val_dataloader() and dm.test_dataloader()

    num_datapoints = len(dm.data_train) + len(dm.data_val) + len(dm.data_test)
    assert num_datapoints == 101

    batch = next(iter(dm.train_dataloader()))
    x = batch
    assert isinstance(x,dict)
    k = list(x.keys())
    assert k == ['audio','loudness','f0','rms']