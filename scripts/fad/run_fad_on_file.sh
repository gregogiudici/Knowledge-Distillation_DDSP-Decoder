#! /bin/bash

## run_fad_on_file.sh
## Computes distribution on a specific file and then computes
## distance between that distribution and a background one.
##      Usage: run_fad_on_file.sh [PATH TO FILE] [EXPERIMENT ID] [BACKGROUND STATS] [FILENAME TO SAVE RESULTS TO]

FILE_PATH=$1
STAT_NAME=$2
BKG_STAT=$3
RESULTS_FILENAME=$4

source scripts/fad/run_stats_on_file.sh ${FILE_PATH} ${STAT_NAME}
source scripts/fad/run_fad.sh ${STAT_NAME} ${BKG_STAT} ${RESULTS_FILENAME}
