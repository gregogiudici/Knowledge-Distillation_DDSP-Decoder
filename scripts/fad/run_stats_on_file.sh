#! /bin/bash

## run_stats_on_file.sh
## Computes distribution on a specific file.
##      Usage: run_stats_on_file.sh [PATH TO FILE] [EXPERIMENT ID]

FILE_PATH=$1
STAT_NAME=$2

mkdir $STATS_DIR -p

# Store list
ls -d --color=never ${FILE_PATH} >${STATS_DIR}/${STAT_NAME}.list

# Compute Stats
STATS_FILE=${STATS_DIR}/${STAT_NAME}.stats
if test -f "$STATS_FILE"; then
echo "FILE STATS: ${STAT_NAME} already exists"
else
cd ${GOOGLE_RES}
python -m frechet_audio_distance.create_embeddings_main \
    --input_files ${STATS_DIR}/${STAT_NAME}.list \
    --stats ${STATS_FILE}

cd ${ROOT}
fi