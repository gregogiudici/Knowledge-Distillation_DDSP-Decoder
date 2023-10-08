#! /bin/bash

## run_stats_on_urmp.sh
## Computes background distribution on dataset.
##      Usage: run_stats_on_urmp.sh [DATASET DIR] [SAMPLE RATE]

# Step1: Get background distributions (stats) from the whole dataset.

# Background Distribution (copied files from create_data.py)
DATA_DIR=$1
SAMPLE_RATE=$2

mkdir $BKG_DIR -p

for instrument in violin trumpet flute
do
    ls -d --color=never ${DATA_DIR}/${instrument}/*.wav > \
        ${BKG_DIR}/${instrument}_background_${SAMPLE_RATE}.list
done

cd ${GOOGLE_RES}

for instrument in flute violin trumpet
do
    STATS_FILE=${BKG_DIR}/${instrument}_background_${SAMPLE_RATE}.stats
    if test -f "$STATS_FILE"; then
    echo "URMP STATS: ${instrument} ${SAMPLE_RATE} background exists."
    else
    python -m frechet_audio_distance.create_embeddings_main \
        --input_files ${BKG_DIR}/${instrument}_background_${SAMPLE_RATE}.list \
        --stats ${STATS_FILE}
    fi
done
cd ${ROOT}