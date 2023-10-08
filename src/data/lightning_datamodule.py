from typing import Any, Dict, Optional, Tuple

import torch
from lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from src.data.h5_dataset import h5Dataset
from src.data.preprocessor import F0LoudnessRMSPreprocessor
import hydra


class DecoderDataModule(LightningDataModule):
    """LightningDataModule for URMP dataset.

    A DataModule implements 6 key methods:
        def prepare_data(self):
            # things to do on 1 GPU/TPU (not on every GPU/TPU in DDP)
            # download data, pre-process, split, save to disk, etc...
        def setup(self, stage):
            # things to do on every process in DDP
            # load data, set variables, etc...
        def train_dataloader(self):
            # return train dataloader
        def val_dataloader(self):
            # return validation dataloader
        def test_dataloader(self):
            # return test dataloader
        def teardown(self):
            # called on every process in DDP
            # clean up after fit or test

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://lightning.ai/docs/pytorch/latest/data/datamodule.html
    """

    def __init__(
        self,
        data_dir: str = "dataset/data",
        instrument: str = "flute",
        sample_rate: int = 16000,
        frame_rate: int = 250,
        audio_length: int = 4,
        train_split = 0.80,
        batch_size: int = 16,
        input_keys: tuple = ('audio','loudness','f0','rms'),
        seed = 42,
        num_workers: int = 0,
        pin_memory: bool = False,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)
        self.sr = sample_rate
        self.train_split = train_split
         
        self.input_keys = input_keys
        self.traindata_path = '{}'.format(f'{data_dir}/train/{instrument}/{sample_rate}sr_{frame_rate}fr_{audio_length}s.h5')
        
        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None
        self.seed = seed

    @property
    def get_keys_(self):
        return self.input_keys

    def prepare_data(self):
        """Download data if needed.

        Do not use it to assign state (self.x = y).
        """
        pass

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning with both `trainer.fit()` and `trainer.test()`, so be
        careful not to execute things like random split twice!
        """
        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
                dataset = h5Dataset(sr=self.sr,
                    data_path=self.traindata_path,
                    input_keys=self.input_keys,
                    max_audio_val=1
                    )

                train_split = int( self.train_split * len(dataset) )
                test_split = ( len(dataset) - train_split ) // 2
                val_split = len(dataset)- train_split - test_split
                self.data_train, self.data_val, self.data_test = random_split(
                    dataset,
                    [train_split,val_split,test_split],
                    generator=torch.Generator().manual_seed(self.seed)
                )
    
    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def teardown(self, stage: Optional[str] = None):
        """Clean up after fit or test."""
        pass

    def state_dict(self):
        """Extra things to save to checkpoint."""
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Things to do when loading checkpoint."""
        pass


if __name__ == "__main__":
    _ = DecoderDataModule()
