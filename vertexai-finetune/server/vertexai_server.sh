SCRIPT_DIR=$(dirname "$(realpath "$0")")

if [ -f $SCRIPT_DIR/.env ]; then
    set -a  # automatically export all variables
    source $SCRIPT_DIR/.env
    set +a
fi

echo HOST = $HOST

litellm --config $SCRIPT_DIR/config.yaml --host 0.0.0.0 --port $(cat $SCRIPT_DIR/port.txt) &> $SCRIPT_DIR/slurm_subprocess_$SLURM_JOB_ID.log
