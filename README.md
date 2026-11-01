# Sparse-view 3D vessel reconstruction

DiffCond reconstructs 3D vessels using visual-hull initialization, a diffusion
prior, and projection consistency. This repository includes the model code,
pretrained diffusion and segmentation weights, and a reconstruction example.

## Install

Use an NVIDIA GPU with a compatible CUDA driver.

```bash
conda env create -f environment.yml
conda activate sparse-vessel-reconstruction
```

## Run

The example generates simulated X-rays from your CT volume and matching vessel
mask, segments the projections, and reconstructs the vessel volume. Supply
`1.img.nii.gz` in the CT folder and `1_left.label.nii.gz` or
`1_right.label.nii.gz` in the mask folder. Replace `1` with your numeric case ID.
Include only one side's mask to reconstruct one coronary tree.

```bash
DATA_DIR=/path/to/vessel_masks \
RAW_ROOT=/path/to/ct_volumes \
CASE_ID=1 \
OUTPUT_DIR=outputs/example \
CONTRAST=0.004 N_VIEWS=4 \
bash scripts/reconstruct_one_case.sh
```

The script uses the bundled checkpoints and saves reconstruction volumes and
visualizations under `OUTPUT_DIR`. It generates one stochastic reconstruction
per tree. Set `SEED` to change the sample; supported contrasts are `0.004` and
`0.008`, with `N_VIEWS` set to `2`, `4`, or `8`.

For additional options:

```bash
python scripts/reconstruct_conditional_diffusion.py --help
```

Model definitions are in [src/diffusion_model.py](src/diffusion_model.py) and
[src/segmentation_model.py](src/segmentation_model.py). See
[checkpoints](checkpoints/) for weights and [third-party notices](THIRD_PARTY_NOTICES.md)
for dependencies.
