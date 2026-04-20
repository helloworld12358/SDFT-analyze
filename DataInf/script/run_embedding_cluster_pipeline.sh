#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PHASE="${EMBED_CLUSTER_PHASE:-stage1}"

case "$PHASE" in
  stage1)
    bash "$SCRIPT_DIR/run_embedding_cluster_stage1_scan.sh" "$@"
    ;;
  stage2)
    bash "$SCRIPT_DIR/run_embedding_cluster_stage2_select_layers.sh" "$@"
    ;;
  stage3)
    bash "$SCRIPT_DIR/run_embedding_cluster_stage3_tsne.sh" "$@"
    ;;
  full_serial)
    bash "$SCRIPT_DIR/run_embedding_cluster_stage1_scan.sh" "$@"
    bash "$SCRIPT_DIR/run_embedding_cluster_stage2_select_layers.sh" "$@"
    bash "$SCRIPT_DIR/run_embedding_cluster_stage3_tsne.sh" "$@"
    ;;
  *)
    echo "Unknown EMBED_CLUSTER_PHASE=$PHASE"
    echo "Supported phases: stage1, stage2, stage3, full_serial"
    exit 2
    ;;
esac

