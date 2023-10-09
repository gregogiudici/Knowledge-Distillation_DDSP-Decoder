# Metrics for model validation

## 1. [Frechét Audio Distance](https://arxiv.org/abs/1812.08466)
Unlike existing audio evaluation metrics, FAD does not look at individual audio clips, but
instead compares embedding statistics generated on the whole evaluation set with embedding
statistics generated on a large set of clean music (e.g. the training set). This makes FAD a
reference-free metric which can be used to score an an evaluation set where the ground
truth reference audio is not available.

**How to compute:** see [fad](../scripts/fad/) scripts

