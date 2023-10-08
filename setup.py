#!/usr/bin/env python

from setuptools import find_packages, setup

setup(
    name="kd_ddsp_decoder",
    version="0.0.1",
    description="Knowledge Distillation of neural models for audio signal generation",
    author="Gregorio Andrea Giudici",
    author_email="greg.giudici96@gmail.com",
    url="https://github.com/gregogiudici",
    install_requires=["pytorch-lightning", "hydra-core"],
    packages=find_packages(),
    # use this to customize global commands available in the terminal after installing the package
    entry_points={
        "console_scripts": [
            "train_command = src.train:main",
            "eval_command = src.eval:main",
        ]
    },
)
