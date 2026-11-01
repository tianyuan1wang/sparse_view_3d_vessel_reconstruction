#!/usr/bin/env bash
# Generate one stochastic reconstruction per matching coronary tree.
# CASE_ID selects a numeric source CT, so both sides run if both masks are in
# DATA_DIR. Put just one tree's label in DATA_DIR to run exactly one tree.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

: "${DATA_DIR:?Set DATA_DIR to the directory containing split vessel masks}"
: "${RAW_ROOT:?Set RAW_ROOT to the flat directory containing matching CT volumes}"
DIFFUSION_CKPT="${DIFFUSION_CKPT:-${PROJECT_ROOT}/checkpoints/diffusion_model.pt}"
SEGMENTATION_CKPT="${SEGMENTATION_CKPT:-${PROJECT_ROOT}/checkpoints/segmentation_model.pt}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR for generated reconstructions}"
: "${CASE_ID:?Set CASE_ID to the numeric anatomy identifier}"

PYTHON_BIN="${PYTHON_BIN:-python}"
CONTRAST="${CONTRAST:-0.004}"
N_VIEWS="${N_VIEWS:-4}"
case "${CONTRAST}" in
  0.004) DEFAULT_T0=20; DEFAULT_THRESHOLD=0.05 ;;
  0.008) DEFAULT_T0=10; DEFAULT_THRESHOLD=0.15 ;;
  *) echo "Use CONTRAST=0.004 or CONTRAST=0.008" >&2; exit 2 ;;
esac
T0="${T0:-${DEFAULT_T0}}"
THRESHOLD="${THRESHOLD:-${DEFAULT_THRESHOLD}}"
SEED="${SEED:-0}"
GUIDANCE_MODE="${GUIDANCE_MODE:-full}"

export MPLBACKEND="${MPLBACKEND:-Agg}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${TMPDIR:-/tmp}/sparse-vessel-mpl}"
mkdir -p "${MPLCONFIGDIR}" "${OUTPUT_DIR}"

"${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/reconstruct_conditional_diffusion.py" \
  --data_dir "${DATA_DIR}" \
  --raw_roots "${RAW_ROOT}" \
  --out_dir "${OUTPUT_DIR}" \
  --case_id "${CASE_ID}" \
  --angle_list "${N_VIEWS}" \
  --volume_grid_shape 128 128 128 \
  --mask_preprocessing training_matched \
  --volume_mask_resample_mode nearest \
  --R_SO_MM 400 \
  --PLANE yz \
  --TILT_DEG 0 \
  --TILT_AXIS x \
  --d_OD_mm 400 \
  --det_h 256 \
  --det_w 256 \
  --det_h_mm 300 \
  --det_w_mm 300 \
  --bbox_pad 0 \
  --min_vox_component 50 \
  --device cuda \
  --use_trained_segmentation \
  --soft_seg_ckpt "${SEGMENTATION_CKPT}" \
  --soft_seg_device cuda \
  --soft_seg_base_ch 32 \
  --soft_seg_pred_thr 0.5 \
  --xray_photon_count 10000 \
  --xray_hu_scale 0.003 \
  --xray_hu_min -1000 \
  --xray_hu_max 3000 \
  --xray_delta_mu "${CONTRAST}" \
  --imperfect_severity 0 \
  --seg_corrupt_view_prob 0 \
  --diffusion_prior_ckpt "${DIFFUSION_CKPT}" \
  --diffusion_prior_device cuda \
  --recon_mode cond_diffusion \
  --guidance_mode "${GUIDANCE_MODE}" \
  --pnp_init_mode vh \
  --cond_init_method q_sample \
  --cond_vh_soft_max 0.30 \
  --diffusion_denoise_t "${T0}" \
  --diffusion_n_averages 1 \
  --cond_dc_step_size 0.03 \
  --lambda_tv_cond 0 \
  --guidance_inner_steps 1 \
  --diffusion_dc_loss_type bce \
  --alpha_mode neglog \
  --reconstruction_seed "${SEED}" \
  --eval_threshold_mode fixed \
  --eval_fixed_thr "${THRESHOLD}"
