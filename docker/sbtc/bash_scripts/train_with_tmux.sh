#!/bin/bash

echo ""
echo "Training is being started in a tmux sesion..."


# Ensure TMUX_SCRIPT_DIRECTORY is set, default to the current directory if not.
: ${TMUX_SCRIPT_DIRECTORY:=$(pwd)}
echo "TMUX_SCRIPT_DIRECTORY is set to: ${TMUX_SCRIPT_DIRECTORY}"

# Default value for LIBRARY.
LIBRARY="rls_rl"

# Default arguments for rls_rl (and skrl)
default_args=(--task "SBTC-Unscrew-Franka-OSC-v0" --headless --livestream 0)

# Array to hold additional (user-supplied) arguments.
user_args=()

# Parse command-line arguments.
# Accept flags: --rsl_rl, --skrl, --eureka.
# All other arguments will be collected into user_args.
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
            # Collect any other parameter into user_args.
            user_args+=("$1")
            shift
            ;;
    esac
done

# Determine the Python script and final ARGS based on the selected library.
if [ "$LIBRARY" = "eureka" ]; then
    PYTHON_SCRIPT="${TMUX_SCRIPT_DIRECTORY}/_isaaclab_eureka/scripts/train.py"
    # For eureka, no default arguments—only user-supplied ones.
    final_args=("${user_args[@]}")
elif [ "$LIBRARY" = "skrl" ]; then
    PYTHON_SCRIPT="${TMUX_SCRIPT_DIRECTORY}/scripts/reinforcement_learning/skrl/train.py"
    # Merge default_args with any user-supplied args.
    final_args=("${default_args[@]}" "${user_args[@]}")
elif [ "$LIBRARY" = "rls_rl" ]; then
    PYTHON_SCRIPT="${TMUX_SCRIPT_DIRECTORY}/scripts/reinforcement_learning/rsl_rl/train.py"
    final_args=("${default_args[@]}" "${user_args[@]}")
else
    echo "Warning: Library '$LIBRARY' is not recognized. Defaulting to 'rls_rl'."
    PYTHON_SCRIPT="${TMUX_SCRIPT_DIRECTORY}/scripts/reinforcement_learning/rsl_rl/train.py"
    final_args=("${default_args[@]}" "${user_args[@]}")
fi

# Print the command settings for verification.
echo "Using library: ${LIBRARY}"
echo "Using Python script: ${PYTHON_SCRIPT}"
echo "With arguments: ${final_args[@]}"

# Configurable variables for tmux/session handling.
SESSION_NAME="isaaclab_training"
LOG_DIR="${TMUX_SCRIPT_DIRECTORY}/tmux"
TIMESTAMP=$(date '+%Y-%m-%d_%H-%M-%S')
LOG_FILE="${LOG_DIR}/${TIMESTAMP}_${LIBRARY}.log"
TRAIN_CMD="cd ${TMUX_SCRIPT_DIRECTORY} && ${TMUX_SCRIPT_DIRECTORY}/_isaac_sim/python.sh ${PYTHON_SCRIPT} ${final_args[@]} | tee -a ${LOG_FILE}"

# Ensure the log directory exists.
mkdir -p "${LOG_DIR}"

sleep 3  # Optional sleep to let the user see the output before proceeding.

# --- CASE 1: Running inside a tmux session ---
if [ -n "$TMUX" ]; then
    current_session=$(tmux display-message -p '#S')
    if [ "$current_session" != "${SESSION_NAME}" ]; then
        echo "You are inside tmux session '$current_session', not the target '$SESSION_NAME'."
        echo "Detaching from the current session and re-running the script without TMUX..."
        tmux detach-client
        # Re-run the script with TMUX unset so that subsequent logic treats us as outside tmux.
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