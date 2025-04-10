#!/bin/bash

echo ""
echo "Training is being started in a tmux session..."

# Ensure TMUX_SCRIPT_DIRECTORY is set, default to the current directory if not.
: ${TMUX_SCRIPT_DIRECTORY:=$(pwd)}
echo "TMUX_SCRIPT_DIRECTORY is set to: ${TMUX_SCRIPT_DIRECTORY}"

###############################################
# Default settings for library and workflow.
###############################################
# The workflow: "isaaclab" (use library training script) or "eureka" (use eureka training script)
WORKFLOW="isaaclab"

# The library (only rsl_rl and skrl are allowed; default is rsl_rl)
LIBRARY="rsl_rl"

# Default arguments for the isaaclab workflow (for rsl_rl and skrl)
default_args=(--task "SBTC-Unscrew-Franka-OSC-v0" --headless --livestream 0)

# Array to hold additional (user-supplied) arguments.
user_args=()

###############################################
# Parse command-line arguments.
###############################################
while [[ $# -gt 0 ]]; do
    case "$1" in
        # Library selection
        --library)
            shift
            LIBRARY="$1"
            shift
            ;;
        --rsl_rl)
            LIBRARY="rsl_rl"
            shift
            ;;
        --skrl)
            LIBRARY="skrl"
            shift
            ;;
        # Workflow selection
        --workflow)
            shift
            WORKFLOW="$1"
            shift
            ;;
        --eureka)
            WORKFLOW="eureka"
            shift
            ;;
        # Future options (like --ray) can be added here.
        *)
            # Collect any other parameter into user_args.
            user_args+=("$1")
            shift
            ;;
    esac
done

###############################################
# Library sanity check and assignment of LIBRARY_SCRIPT.
###############################################
if [ "$LIBRARY" = "rsl_rl" ]; then
    LIBRARY_SCRIPT="${TMUX_SCRIPT_DIRECTORY}/scripts/reinforcement_learning/rsl_rl/train.py"
elif [ "$LIBRARY" = "skrl" ]; then
    LIBRARY_SCRIPT="${TMUX_SCRIPT_DIRECTORY}/scripts/reinforcement_learning/skrl/train.py"
else
    echo "WARNING: Library '$LIBRARY' is not recognized. Defaulting to 'rsl_rl'."
    LIBRARY_SCRIPT="${TMUX_SCRIPT_DIRECTORY}/scripts/reinforcement_learning/rsl_rl/train.py"
    LIBRARY="rsl_rl"
fi

###############################################
# Workflow sanity check and TRAINING_SCRIPT assignment.
###############################################
if [ "$WORKFLOW" = "eureka" ]; then
    TRAINING_SCRIPT="${TMUX_SCRIPT_DIRECTORY}/_isaaclab_eureka/scripts/train.py"
    # For eureka, do not prepend default arguments.
    final_args=("${user_args[@]}")
elif [ "$WORKFLOW" = "isaaclab" ]; then
    TRAINING_SCRIPT="${LIBRARY_SCRIPT}"
    final_args=("${default_args[@]}" "${user_args[@]}")
else
    echo "WARNING: Workflow '$WORKFLOW' is not recognized. Defaulting to 'isaaclab'."
    WORKFLOW="isaaclab"
    TRAINING_SCRIPT="${LIBRARY_SCRIPT}"
    final_args=("${default_args[@]}" "${user_args[@]}")
fi

###############################################
# Print the command settings for verification.
###############################################
echo "Selected library: ${LIBRARY}"
echo "Using workflow: ${WORKFLOW}"
echo "Using training script: ${TRAINING_SCRIPT}"
echo "With arguments: ${final_args[@]}"

###############################################
# TMUX session and logging configuration.
###############################################
if [ "$WORKFLOW" = "eureka" ]; then
    SESSION_NAME="eureka_training"
else
    SESSION_NAME="isaaclab_training"
fi

LOG_DIR="${TMUX_SCRIPT_DIRECTORY}/tmux"
TIMESTAMP=$(date '+%Y-%m-%d_%H-%M-%S')
LOG_FILE="${LOG_DIR}/${WORKFLOW}/${TIMESTAMP}_${LIBRARY}.log"

# Ensure the log directory exists (including the workflow subdirectory).
mkdir -p "${LOG_DIR}/${WORKFLOW}"

TRAIN_CMD="cd ${TMUX_SCRIPT_DIRECTORY} && ${TMUX_SCRIPT_DIRECTORY}/_isaac_sim/python.sh ${TRAINING_SCRIPT} ${final_args[@]} | tee -a ${LOG_FILE}"

sleep 3  # Optional sleep to let the user see the output before proceeding.

###############################################
# --- CASE 1: Running inside a tmux session ---
###############################################
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

###############################################
# --- CASE 2 & 3: Running outside any tmux session ---
###############################################
if ! tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
    echo "Target tmux session '$SESSION_NAME' does not exist. Creating new session..."
    tmux new-session -d -s "${SESSION_NAME}" -n training "${TRAIN_CMD}; bash"
else
    echo "Target tmux session '$SESSION_NAME' already exists."
    tmux send-keys -t "${SESSION_NAME}:training" "$TRAIN_CMD" C-m
fi

# Finally, attach to the target session.
tmux attach-session -t "${SESSION_NAME}"