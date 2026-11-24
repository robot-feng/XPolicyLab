#!/bin/bash
# Resolve EVAL_ENV_TYPE to sim, debug, robomme, or real_world.
# Empty or unset defaults to sim.

resolve_eval_env_type() {
    local raw="${EVAL_ENV_TYPE:-}"
    case "${raw}" in
        ""|sim)
            echo "sim"
            ;;
        debug)
            echo "debug"
            ;;
        robomme)
            echo "robomme"
            ;;
        real|real_world)
            echo "real_world"
            ;;
        *)
            echo "[ERROR] Unknown EVAL_ENV_TYPE: '${raw}' (expected: sim, debug, robomme, real)" >&2
            return 1
            ;;
    esac
}
