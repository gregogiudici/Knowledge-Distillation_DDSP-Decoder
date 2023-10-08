#! /bin/bash
source .env

# URMP COPIED FILES, RESAMPLED AT 16kHz
# This will create a directory called 'background_stats'
# With three background stats:
#   1. violin_background_16k
#   2. flute_background_16k
#   3. trumpet_background_16k

source scripts/fad/run_stats_on_urmp.sh ${URMP_DIR_16} 16000

# URMP COPIED FILES, AS IS, 48Khz
# This will create a directory called 'background_stats'
# With three background stats:
#   1. violin_background_48k
#   2. flute_background_48k
#   3. trumpet_background_48k

#source scripts/fad/run_stats_on_urmp.sh ${URMP_DIR_48} 48000