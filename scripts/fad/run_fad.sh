#! bin/bash

## run_fad.sh
## Computes distance between two distributions
##      Usage: run_fad.sh [EXPERIMENT ID] [BACKGROUND STATS] [FILENAME TO SAVE RESULTS TO]

STAT_NAME=$1
BKG_STAT=$2
RESULTS_FILENAME=$3

cd ${GOOGLE_RES}
FAD=$(python -m frechet_audio_distance.compute_fad \
    --background_stats ${BKG_DIR}/${BKG_STAT}.stats \
    --test_stats ${STATS_DIR}/${STAT_NAME}.stats | grep FAD)

cd ${FAD_DIR}
echo ${STAT_NAME} $FAD >>${RESULTS_FILENAME}
echo "FAD = $FAD"
cd ${ROOT}