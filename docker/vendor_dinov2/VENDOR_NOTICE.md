# Vendored third-party code — DINOv2

`vendor_dinov2/` is a copy of [facebookresearch/dinov2](https://github.com/facebookresearch/dinov2),
taken 2026-08-14. It is vendored rather than installed because the challenge's
evaluation container has no network access at runtime; the Dockerfile points
`FUBIO_DINOV2_LOCAL` at this directory so `torch.hub` resolves offline.

The tree is unmodified. The exact upstream commit was not recorded at clone
time; the copy corresponds to the state of `main` on 2026-08-14, which includes
the Cell-DINO (2025-12-16) and XRay-DINO (2025-12-18) additions.

## What we actually use

Only `dinov2_vitb14` — the original Meta AI DINOv2 ViT-B/14 backbone, released
under the **Apache License 2.0** (`LICENSE`). Nothing else in this tree is
loaded at training or inference time.

## Licences present in this tree, and why they are kept

Upstream ships several licences because later contributions added models under
separate terms:

| File | Covers | Used by us |
|---|---|---|
| `LICENSE` | DINOv2 code and the original DINOv2 backbones (Apache-2.0) | **Yes** — `dinov2_vitb14` |
| `LICENSE_CELL_DINO_CODE` | Cell-DINO code (Creative Commons) | No |
| `LICENSE_CELL_DINO_MODELS` | Cell-DINO weights (FAIR Noncommercial Research) | No |
| `LICENSE_XRAY_DINO_MODEL` | XRay-DINO weights (X-Ray DINO Research License) | No |

All four are retained deliberately. Apache-2.0 §4 requires that redistribution
preserve the licence and notices received with the work — stripping the unused
licence files would be a licence violation, not a tidy-up. Their presence does
**not** place any non-commercial restriction on this project: those terms attach
to model weights we never download or load.

The FU_Biometry challenge permits only publicly available pretrained models;
`dinov2_vitb14` qualifies, and its Apache-2.0 terms are compatible with this
repository's MIT licence.
