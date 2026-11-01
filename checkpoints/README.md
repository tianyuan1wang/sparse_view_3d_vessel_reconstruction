# Pretrained weights

- `diffusion_model.pt`: 3D diffusion prior, including the inference configuration.
- `segmentation_model.pt`: 2D vessel segmenter (`UNet2DSoftSeg`, `base_ch=32`).

Both checkpoints are included and used by the reconstruction example. They
contain model weights without optimizer state or training-data paths.
Verify downloads with `cd checkpoints && sha256sum -c SHA256SUMS`.
