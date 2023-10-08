# Fréchet Audio Distance

1. **Create Conda Environment:** in project (root) directory run
```bash
conda deactivate
conda env create -n fad
conda activate fad
pip install apache-beam numpy scipy tensorflow tf-slim
```
2. **Clone [google-research](https://github.com/google-research/google-research/tree/master/frechet_audio_distance):** in another directory run
```bash
git clone https://github.com/google-research/google-research.git
cd google-research
mkdir tensorflow_models
touch tensorflow_models/__init__.py
svn export https://github.com/tensorflow/models/trunk/research/audioset tensorflow_models/audioset/
touch tensorflow_models/audioset/__init__.py
```
**NOTE**: there are issues, fix them following [this](https://github.com/google-research/google-research/issues/227)

2. **Add google_research to project path:** add google_research to your **.env** file
```bash
# EXAMPLE
GOOGLE_RES="/path/to/google-research"
```

3. **Resample dataset to 16kHz**: in [dataset](../../dataset/) directory run 
```bash
mkdir -p files/train16/{flute,trumpet,violin}
python -m resample.py files/train/flute files/train16/flute
python -m resample.py files/train/trumpet files/train16/trumpet
python -m resample.py files/train/violin files/train16/violin
```

4. **Compute background:** in project (root) directory run
```bash
source scripts/fad/compute_background.sh
```

5. **Compute Frechét Audio Distance:** in project (root) directory run
```bash
source scripts/fad/compute_ddsp.sh [DIMENSION] [DECODER] [INSTRUMENT] [RESULTS_FILE_NAME=INSTRUMENT]

# e.g. source scripts/fad/compute_ddsp.sh large gru flute
```

6. Check results in [flute.fad](../../metrics/frechet_audio_distance/)