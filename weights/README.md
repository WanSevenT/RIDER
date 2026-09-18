# Pretrained weights

This directory contains the compact checkpoints used for the reported RIDER model.

Load them in this exact order:

1. Base OpenAI CLIP `ViT-L/14`
2. `semantic_best.pth`
3. `artifact_best.pth`
4. `fusion_best.pth`

`fusion_best.pth` must be applied last because it contains the artifact BatchNorm running buffers updated during Phase II in addition to the fusion-specific parameters.

Checkpoint metadata records:

- semantic expert: epoch 5, CLIP final visual block + LayerNorm, patch shuffle disabled;
- artifact expert: epoch 20, spectral magnitude + Haar wavelet + MS-NPR, image size 224, NPR scales 0.25/0.5/0.75;
- fusion/router: epoch 4 selected from training up to 20 epochs.

Verify the files with:

```bash
(cd weights && sha256sum -c checksums.sha256)
```
