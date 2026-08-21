#!/usr/bin/env bash
#
# Generate the SVD-truncated MegaLoc checkpoints published as release assets.
#
# Forges the pretrained 8448-D MegaLoc model down to each target descriptor
# dimension with truncated SVD of the final linear projection, writing one
# portable state_dict per dimension. The source model is loaded through the
# `megaloc` registry factory, which downloads the pretrained checkpoint and
# requires CUDA; the truncation itself is device-agnostic and the CLI moves
# each forged model to CPU before writing.
#
# The outputs are the assets for the `megaloc-svd-truncated-v1.0` release. The
# revision here stays in lockstep with the release tag and the `-v1.0` filename
# suffix; bump both together for a future revision.
#
# Usage:
#   scripts/publish_megaloc_svd_checkpoints.sh [OUTPUT_DIR]
#
# OUTPUT_DIR defaults to the current directory.

set -euo pipefail

readonly MODEL_KEY="megaloc"
readonly METHOD="svd"
readonly REVISION="1.0"
readonly DIMS=(256 512 1024)

output_dir="${1:-.}"
mkdir -p "${output_dir}"

for dim in "${DIMS[@]}"; do
    output_path="${output_dir}/${MODEL_KEY}-${dim}d-${METHOD}-truncated-v${REVISION}.pth"
    echo "Forging ${MODEL_KEY} to ${dim}d -> ${output_path}"
    uv run viper forge adapt "${MODEL_KEY}" \
        --method "${METHOD}" \
        --dim "${dim}" \
        --revision "${REVISION}" \
        --output "${output_path}"
done

echo "Done. Wrote ${#DIMS[@]} checkpoints to ${output_dir}"
