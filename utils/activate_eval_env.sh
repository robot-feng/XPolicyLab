#!/usr/bin/env bash
# Select an evaluator interpreter without requiring Conda for uv/venv installs.
activate_eval_env() {
    local requested_env="${1:?evaluation environment required}"
    if [[ -d "${requested_env}" && -x "${requested_env}/bin/python" && ! -d "${requested_env}/conda-meta" ]]; then
        if [[ -f "${requested_env}/bin/activate" ]]; then
            source "${requested_env}/bin/activate"
        fi
        XPL_EVAL_PYTHON="${requested_env}/bin/python"
    elif [[ -f "${requested_env}" && -x "${requested_env}" ]]; then
        XPL_EVAL_PYTHON="${requested_env}"
    else
        if ! command -v conda >/dev/null 2>&1; then
            echo "[ERROR] Evaluation environment is not an executable/prefix and Conda is unavailable: ${requested_env}" >&2
            return 1
        fi
        source "$(conda info --base)/etc/profile.d/conda.sh"
        conda activate "${requested_env}" || return 1
        XPL_EVAL_PYTHON="$(command -v python)"
    fi
    export XPL_EVAL_PYTHON
}
