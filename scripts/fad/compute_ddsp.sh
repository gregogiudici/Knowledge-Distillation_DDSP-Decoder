#! /bin/bash
source .env
export TF_ENABLE_ONEDNN_OPTS=0
# Run FAD to compare reference and resynthesis of ddsp_decoder
# Usage: compute_ddsp.sh [DIMENSION] [DECODER] [INSTRUMENT] [RESULTS FILE=results.fad]
DIMENSION=$1
DECODER=$2
INSTRUMENT=$3
RESULT_FILE="${4:-$3.fad}"

SR="16000"
SR_FR="16000_250"

case $DIMENSION in

  large)
    ;;

  medium)
    ;;

  small)
    ;;

  *)
    echo "Invalid [DIMENSION]! Must be: large , medium , small"
    echo "Usage: compute_ddsp.sh [DIMENSION] [DECODER] [INSTRUMENT] [RESULTS FILE=results.fad]"
    return
    ;;

esac


case $DECODER in

  gru)
    KD=0;
    echo $KD
    MODEL_PATH=${DIMENSION^^}_${DECODER^^}_${INSTRUMENT^^}_${SR_FR};
    EXPERIMENT_NAME=${DIMENSION}_${DECODER}_${INSTRUMENT}_${SR_FR};
    ;;

  tcn)
    KD=0
    MODEL_PATH=${DIMENSION^^}_${DECODER^^}_${INSTRUMENT^^}_${SR_FR}
    EXPERIMENT_NAME=${DIMENSION}_${DECODER}_${INSTRUMENT}_${SR_FR}
    ;;

  s4)
    KD=0
    MODEL_PATH=${DIMENSION^^}_${DECODER^^}_${INSTRUMENT^^}_${SR_FR}
    EXPERIMENT_NAME=${DIMENSION}_${DECODER}_${INSTRUMENT}_${SR_FR}
    ;;

  ddx7)
    KD=0
    MODEL_PATH=${DECODER^^}_${INSTRUMENT^^}_${SR_FR}
    EXPERIMENT_NAME=${DECODER}_${INSTRUMENT}_${SR_FR}
    ;;

  gru2gru)
    KD=1
    MODEL_PATH=${DIMENSION^^}_${DECODER^^}_${INSTRUMENT^^}_${SR_FR}
    EXPERIMENT_NAME=${DIMENSION}_${DECODER}_${INSTRUMENT}_${SR_FR}
    ;;

  tcn2tcn)
    KD=1
    MODEL_PATH=${DIMENSION^^}_${DECODER^^}_${INSTRUMENT^^}_${SR_FR}
    EXPERIMENT_NAME=${DIMENSION}_${DECODER}_${INSTRUMENT}_${SR_FR}
    ;;

  s42s4)
    MODEL_PATH=${DIMENSION^^}_${DECODER^^}_${INSTRUMENT^^}_${SR_FR}
    EXPERIMENT_NAME=${DIMENSION}_${DECODER}_${INSTRUMENT}_${SR_FR}
    KD=1
    ;;

  gru2ddx7)
    MODEL_PATH=${DECODER^^}_${INSTRUMENT^^}_${SR_FR}
    EXPERIMENT_NAME=${DECODER}_${INSTRUMENT}_${SR_FR}
    KD=1
    ;;
    
  tcn2ddx7)
    MODEL_PATH=${DECODER^^}_${INSTRUMENT^^}_${SR_FR}
    EXPERIMENT_NAME=${DECODER}_${INSTRUMENT}_${SR_FR}
    KD=1
    ;;

  s42ddx7)
    MODEL_PATH=${DECODER^^}_${INSTRUMENT^^}_${SR_FR}
    EXPERIMENT_NAME=${DECODER}_${INSTRUMENT}_${SR_FR}
    KD=1
    ;;    
  *)
    echo "Invalid [DECODER]! Must be: gru , tcn , s4 , gru2gru , tcn2tcn , s42s4 , gru2ddx7 , tcn2ddx7 , s42ddx7 "
    echo "Usage: compute_ddsp.sh [DIMENSION] [DECODER] [INSTRUMENT] [RESULTS FILE=results.fad]"
    return
    ;;

esac

case $INSTRUMENT in

  flute)
    ;;

  violin)
    ;;

  trumpet)
    ;;

  *)
    echo "Invalid [INSTRUMENT]! Must be flute , trumpet or violin"
    echo "Usage: compute_ddsp.sh [DIMENSION] [DECODER] [INSTRUMENT] [RESULTS FILE=results.fad]"
    return
    ;;

esac


if [ $KD -eq 0 ]; then
  # Run FAD on reference.wav
  # Usage: run_fad_on_file.sh [FILEPATH] [EXP. ID] [BACKGROUND DISTR.] [RESULTS FILE]
  echo "Compute Reference Audio: ${INSTRUMENT^^}_${SR_FR}"
  source scripts/fad/run_fad_on_file.sh \
      ${!MODEL_PATH%/*/*}/results_val_test/${INSTRUMENT}_ref_test.wav \
      ${INSTRUMENT}_ref_${SR} \
      ${INSTRUMENT}_background_${SR} \
      ${RESULT_FILE}

  # Run FAD on resynthesis.wav
  # Usage: run_fad_on_file.sh [FILEPATH] [EXP. ID] [BACKGROUND DISTR.] [RESULTS FILE]
  echo "Compute Synthesized Audio: ${DECODER^^}_${INSTRUMENT^^}_${SR_FR}"

  source scripts/fad/run_fad_on_file.sh \
      ${!MODEL_PATH%/*/*}/results_val_test/${INSTRUMENT}_synth_test.wav \
      ${EXPERIMENT_NAME} \
      ${INSTRUMENT}_background_${SR} \
      ${RESULT_FILE}

elif [ $KD -eq 1 ]; then
  # Run FAD on reference.wav
  # Usage: run_fad_on_file.sh [FILEPATH] [EXP. ID] [BACKGROUND DISTR.] [RESULTS FILE]
  echo "Compute Teacher-Reference Audio: ${INSTRUMENT^^}_${SR_FR}"
  source scripts/fad/run_fad_on_file.sh \
      ${!MODEL_PATH%/*/*}/results_val_test/${INSTRUMENT}_ref_teacher.wav \
      "teacher_${INSTRUMENT}_ref_${SR}" \
      ${INSTRUMENT}_background_${SR} \
      ${RESULT_FILE}

  # Run FAD on resynthesis.wav
  # Usage: run_fad_on_file.sh [FILEPATH] [EXP. ID] [BACKGROUND DISTR.] [RESULTS FILE]
  echo "Compute Teacher-Synthesized Audio: ${DECODER^^}_${INSTRUMENT^^}_${SR_FR}"

  source scripts/fad/run_fad_on_file.sh \
      ${!MODEL_PATH%/*/*}/results_val_test/${INSTRUMENT}_synth_teacher.wav \
      "teacher_${EXPERIMENT_NAME}" \
      ${INSTRUMENT}_background_${SR} \
      ${RESULT_FILE}

    # Run FAD on reference.wav
  # Usage: run_fad_on_file.sh [FILEPATH] [EXP. ID] [BACKGROUND DISTR.] [RESULTS FILE]
  echo "Compute Student-Reference Audio: ${INSTRUMENT^^}_${SR_FR}"
  source scripts/fad/run_fad_on_file.sh \
      ${!MODEL_PATH%/*/*}/results_val_test/${INSTRUMENT}_ref_student.wav \
      "student_${INSTRUMENT}_ref_${SR}" \
      ${INSTRUMENT}_background_${SR} \
      ${RESULT_FILE}

  # Run FAD on resynthesis.wav
  # Usage: run_fad_on_file.sh [FILEPATH] [EXP. ID] [BACKGROUND DISTR.] [RESULTS FILE]
  echo "Compute Student-Synthesized Audio: ${DECODER^^}_${INSTRUMENT^^}_${SR_FR}"

  source scripts/fad/run_fad_on_file.sh \
      ${!MODEL_PATH%/*/*}/results_val_test/${INSTRUMENT}_synth_student.wav \
      "student_${EXPERIMENT_NAME}" \
      ${INSTRUMENT}_background_${SR} \
      ${RESULT_FILE}

  else 
    echo "ERROR: KD variable not defined"
  fi

  
