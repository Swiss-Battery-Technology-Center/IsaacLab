#!/bin/bash

# Ensure TMUX_SCRIPT_DIRECTORY is set, default to the current directory if not.
: ${TMUX_SCRIPT_DIRECTORY:=$(pwd)}
echo "TMUX_SCRIPT_DIRECTORY is set to: ${TMUX_SCRIPT_DIRECTORY}"

# Default value for LIBRARY.
LIBRARY="rls_rl"

# Parse command-line arguments.
while [[ $# -gt 0 ]]; do
    case "$1" in
        --rsl_rl)
            LIBRARY="rls_rl"
            shift
            ;;
        --skrl)
            LIBRARY="skrl"
            shift
            ;;
        --eureka)
            LIBRARY="eureka"
            shift
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

if [ "$LIBRARY" = "eureka" ]; then
    PYTHON_SCRIPT="${TMUX_SCRIPT_DIRECTORY}/_isaaclab_eureka/scripts/train.py"
    ARGS=()
elif [ "$LIBRARY" = "skrl" ]; then
    PYTHON_SCRIPT="${TMUX_SCRIPT_DIRECTORY}/scripts/reinforcement_learning/skrl/train.py"
    ARGS=(--task SBTC-Unscrew-Franka-OSC-v0 --headless --livestream 0)
else
    # Default case (no argument provided or argument is "rls_rl")
    PYTHON_SCRIPT="${TMUX_SCRIPT_DIRECTORY}/scripts/reinforcement_learning/rsl_rl/train.py"
    ARGS=(--task SBTC-Unscrew-Franka-OSC-v0 --headless --livestream 0)
fi

# Print the command that will be executed
echo "Using Python script: $PYTHON_SCRIPT"
echo "With arguments: ${ARGS[@]}"

# Configurable variables
SESSION_NAME="isaaclab_training"
# Note: TMUX_SCRIPT_DIRECTORY is assumed to be already set in the environment
LOG_DIR="${TMUX_SCRIPT_DIRECTORY}/tmux"
LOG_FILE="${LOG_DIR}/${SESSION_NAME}.log"
TRAIN_CMD="cd ${TMUX_SCRIPT_DIRECTORY} && ${TMUX_SCRIPT_DIRECTORY}/_isaac_sim/python.sh ${PYTHON_SCRIPT} ${ARGS[@]} | tee -a ${LOG_FILE}"

# Ensure the log directory exists
mkdir -p "${LOG_DIR}"

sleep 3  # Optional sleep to allow user to see the output before proceeding

# --- CASE 1: Running inside a tmux session ---
if [ -n "$TMUX" ]; then
    current_session=$(tmux display-message -p '#S')
    if [ "$current_session" != "${SESSION_NAME}" ]; then
        echo "You are inside tmux session '$current_session', not the target '$SESSION_NAME'."
        echo "Detaching from the current session and re-running the script without TMUX..."
        tmux detach-client
        # Re-run the script with the TMUX variable removed so that subsequent logic treats us as outside tmux.
        exec env -u TMUX "$0" "$@"
    else
        echo "Already inside the target tmux session '$SESSION_NAME'."
        echo "Running training command in the current pane..."
        eval "$TRAIN_CMD"
        exit 0
    fi
fi

# --- CASE 2 & 3: Running outside any tmux session ---
if ! tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
    echo "Target tmux session '$SESSION_NAME' does not exist. Creating new session..."
    tmux new-session -d -s "${SESSION_NAME}" -n training "${TRAIN_CMD}; bash"
else
    echo "Target tmux session '$SESSION_NAME' already exists."
    tmux send-keys -t "${SESSION_NAME}:training" "$TRAIN_CMD" C-m
fi

# Finally, attach to the target session.
tmux attach-session -t "${SESSION_NAME}"
