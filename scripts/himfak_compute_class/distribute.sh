#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

CONFIG_ONLY=false

# Parse flags
for arg in "$@"; do
    case "$arg" in
        --config-only|-c)
            CONFIG_ONLY=true
            shift
            ;;
    esac
done

NODES="192.168.49.109 192.168.49.111"

REMOTE_DIR="$HOME/opt/PES-Fitting-MSA"

PIP_BASIS="1_2_1_2_4"
CONFIG="models/h2o-h2o/water-extended-66-sharded.xyz"
DATASET="datasets/raw/h2o-h2o/h2o-h2o-extended-10.xyzext"
BASIS_DIR="datasets/external/h2o-h2o"

for NODE in $NODES; do
    echo "==> $NODE"

    if [ "$CONFIG_ONLY" = true ]; then
        ssh "$NODE" "mkdir -p $REMOTE_DIR/$(dirname $CONFIG)"
        scp "$PROJECT_ROOT/$CONFIG" "$NODE:$REMOTE_DIR/$CONFIG"
        continue
    fi

    ssh "$NODE" "mkdir -p $REMOTE_DIR/$(dirname $CONFIG) $REMOTE_DIR/$(dirname $DATASET) $REMOTE_DIR/$BASIS_DIR"

    scp "$PROJECT_ROOT/$CONFIG" "$NODE:$REMOTE_DIR/$CONFIG"

    scp "$PROJECT_ROOT/$DATASET" "$NODE:$REMOTE_DIR/$DATASET"

    scp "$PROJECT_ROOT/$BASIS_DIR"/c_basis_${PIP_BASIS}.cc "$NODE:$REMOTE_DIR/$BASIS_DIR/"
    scp "$PROJECT_ROOT/$BASIS_DIR"/c_basis_${PIP_BASIS}.h "$NODE:$REMOTE_DIR/$BASIS_DIR/"
    scp "$PROJECT_ROOT/$BASIS_DIR"/c_jac_${PIP_BASIS}.cc  "$NODE:$REMOTE_DIR/$BASIS_DIR/"
    scp "$PROJECT_ROOT/$BASIS_DIR"/c_jac_${PIP_BASIS}.h  "$NODE:$REMOTE_DIR/$BASIS_DIR/"

    scp "$PROJECT_ROOT/$BASIS_DIR"/${PIP_BASIS}.MOL "$NODE:$REMOTE_DIR/$BASIS_DIR/${PIP_BASIS}.BAS"
done

echo "Done."
