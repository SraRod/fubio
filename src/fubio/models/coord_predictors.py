"""GeoSimCC coordinate readout (paper Section 2.4) and its geometry helpers.

Interface: forward(inst_feat, lm_feat, memory, spatial_shape, memory_pos,
mem_k, fine) → (coords, heat). Six earlier readout modes (direct, ref_points,
shape_prior, heatmap, simcc, shape_simcc) were retired with their configs;
GeoSimCC is the readout of the submitted model.

Upstream: train/config.py (GeoSimCCCoordConfig), data/shape_prior.py (TaskShapePrior).
Downstream: heads.py (TaskModule owns a coord_predictor).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from fubio.data.shape_prior import TaskShapePrior
from fubio.data.task_registry import PARAMS
from fubio.train.config import GeoSimCCCoordConfig


def measurement_partners(task_id: str, n_keypoints: int) -> list[int]:
    """Per-landmark partner index defining its measurement axis, from PARAMS.

    The clinical parameter a landmark belongs to fixes the direction along which
    its boundary must be found: for a distance the partner is the other
    endpoint, for an ellipse the opposite end of the same axis, for an angle the
    vertex. -1 when a landmark belongs to no 2-point measurement (only the AOP
    vertex, which has no single axis).

    Returns a list of length n_keypoints.
    """
    partner = [-1] * n_keypoints
    for pdef in PARAMS.values():
        if pdef.task_id != task_id:
            continue
        li = pdef.landmark_indices
        if pdef.formula_type == "distance" and len(li) == 2:
            partner[li[0]], partner[li[1]] = li[1], li[0]
        elif pdef.formula_type == "ellipse_circumference" and len(li) == 4:
            partner[li[0]], partner[li[1]] = li[1], li[0]
            partner[li[2]], partner[li[3]] = li[3], li[2]
        elif pdef.formula_type == "angle" and len(li) == 3:
            partner[li[1]], partner[li[2]] = li[0], li[0]
    return partner


def weighted_similarity_fit(
    src: Tensor,
    dst: Tensor,
    weights: Tensor | None = None,
    eps: float = 1e-6,
) -> Tensor:
    """Least-squares 2D similarity (rotation + uniform scale + translation) src → dst.

    Closed form, differentiable, SVD-free. A similarity has det = s² > 0 by
    construction, so reflections cannot appear (unlike an SVD-based Procrustes
    that needs explicit det correction).

    Computes at fp32 or better: the centering / cross-covariance / normalisation
    chain is numerically fragile under bf16, so half precision is upcast — but
    fp64 is preserved, which is what makes gradcheck meaningful.

    Args:
        src: (..., K, 2) source points (e.g. canonical shape).
        dst: (..., K, 2) target points (e.g. geometric coarse estimates).
        weights: (..., K) non-negative per-point weights, or None for uniform.
        eps: floor on the normalising denominator.

    Returns:
        (..., K, 2) — src mapped onto dst by the fitted transform.

    Degenerate cases (K < 2, coincident or collinear src) fall back gracefully:
    the clamped denominator keeps the rotation finite, and the translation term
    still carries the weighted centroid, so the output stays near dst's centre.
    """
    dtype = src.dtype
    calc = torch.float32 if dtype in (torch.bfloat16, torch.float16) else dtype
    src32, dst32 = src.to(calc), dst.to(calc)

    if weights is None:
        w = torch.full(src32.shape[:-1], 1.0, device=src32.device, dtype=calc)
    else:
        w = weights.to(calc).clamp(min=0.0)
    w = w / w.sum(dim=-1, keepdim=True).clamp(min=eps)
    w_ = w.unsqueeze(-1)  # (..., K, 1)

    src_c = src32 - (w_ * src32).sum(dim=-2, keepdim=True)
    dst_c = dst32 - (w_ * dst32).sum(dim=-2, keepdim=True)

    # a = s·cosθ, b = s·sinθ from the weighted cross-covariance.
    a = (w * (src_c * dst_c).sum(-1)).sum(-1)
    b = (w * (src_c[..., 0] * dst_c[..., 1] - src_c[..., 1] * dst_c[..., 0])).sum(-1)
    den = (w * (src_c**2).sum(-1)).sum(-1).clamp(min=eps)
    a, b = (a / den).unsqueeze(-1), (b / den).unsqueeze(-1)

    rot_x = a * src_c[..., 0] - b * src_c[..., 1]
    rot_y = b * src_c[..., 0] + a * src_c[..., 1]
    rotated = torch.stack([rot_x, rot_y], dim=-1)

    out = rotated + (w_ * dst32).sum(dim=-2, keepdim=True)
    return out.to(dtype)


def normalized_grid(hp: int, wp: int, device: torch.device) -> tuple[Tensor, Tensor]:
    """Per-token normalized coords, pixel-center aligned to the [0,1] landmark space.

    Token (row, col) covers input pixels [col·P, (col+1)·P); its center is
    (col+0.5)·P. Normalized x = (col+0.5)·P / (Wp·P) = (col+0.5)/Wp — matches
    the GT convention (aug_pixel / target_size). Loss target and predictor MUST
    share this grid; it is the single source of truth for that convention.

    Returns:
        (xs, ys) each (Hp*Wp,) float, row-major to match memory token order.
    """
    rows = torch.arange(hp, device=device, dtype=torch.float32)
    cols = torch.arange(wp, device=device, dtype=torch.float32)
    gy, gx = torch.meshgrid(rows, cols, indexing="ij")
    xs = ((gx + 0.5) / wp).flatten()
    ys = ((gy + 0.5) / hp).flatten()
    return xs, ys


class GeoSimCCPredictor(nn.Module):
    """Geometry locates, statistics shape, local evidence refines.

    Three stages, each with a single responsibility:

      1. GEO   — query·memory correlation → soft-argmax over the patch grid.
                 Position is READ from grid geometry, so no feature→coordinate
                 map has to be learned. Deliberately COARSE: its job is to land
                 in the right cell, not to be precise. Supervised directly via
                 the returned heat (module.py's lambda_heatmap path) so it can
                 never hide behind a downstream correction — this is the
                 structural fix for the prior-dependence recorded at
                 HeatmapPredictor's prior_dropout comment.
      2. SHAPE — alpha @ canonical_basis gives a Procrustes-aligned (position-
                 and scale-free) deformation; a closed-form similarity fitted
                 FROM the stage-1 points places it. Position therefore comes
                 from the image and shape from statistics, with no overlap.
                 Denoises K independent stage-1 estimates; for K=2 a similarity
                 is exactly determined, so this stage is a no-op there by
                 construction (IVC / FUGC / fetal_femur).
      3. FINE  — bounded offset regressed from a 1D CONTENT-feature profile
                 sampled ALONG the measurement axis on a high-resolution map.
                 Two choices matter here, both from the val_local error
                 decomposition: A4C's long axes are accurate to ~4 deg while its
                 transverse axes scatter by 50-86 deg, i.e. the model finds the
                 chamber but cannot place endpoints on the myocardial wall.
                   - Sampling along the measurement axis, not a square patch: a
                     wall is a 1D boundary, and a square patch averages the
                     crossing away. This turns 2D localisation into 1D edge
                     finding, which is the shape the problem actually has.
                   - Sampling `fine` rather than `memory`: one memory token
                     covers ~50 original px at 224px input, and a myocardial
                     wall is ~10 px. The offset head must see features finer
                     than the structure it is asked to locate.
                 Offsets are predicted in the measurement's own frame (along /
                 perpendicular), which keeps "slide along the wall" separable
                 from "move onto the wall". memory_pos is excluded: positional
                 embeddings are an addressing signal, not local appearance.

    Keys in : inst_feat (B,N,D), lm_feat (B,N,K,D), memory (B,S,D),
              spatial_shape (Hp,Wp), memory_pos (1,S,D),
              mem_k (B,S,D) pre-projected keys, fine (B,C,Hf,Wf) high-res map.
    Keys out: coords (B,N,K,2) in [0,1], heat (B,N,K,S).
    """

    def __init__(
        self,
        d_model: int,
        n_keypoints: int,
        prior: TaskShapePrior,
        tau: float = 1.0,
        window_cells: float = 1.0,
        n_offset_bins: int = 32,
        n_profile: int = 17,
        shape_weight: float = 0.5,
        d_local: int = 32,
        dropout: float = 0.0,
        partner: list[int] | None = None,
        d_fine: int | None = None,
        own_keys: bool = True,
    ) -> None:
        super().__init__()
        if prior.canonical_mean is None or prior.canonical_basis is None:
            raise ValueError(
                "GeoSimCCPredictor requires canonical_mean/canonical_basis — "
                "rebuild shape_prior.json with the current build_shape_prior."
            )
        self._K = n_keypoints
        self._n_prof = n_profile
        self._n_bins = n_offset_bins
        self._window_cells = window_cells
        # Fixed, not learned: a learned fusion weight can simply drive the
        # visual branch to zero authority before it has learned anything.
        self._shape_w = shape_weight

        # --- Stage 1 ---
        self.q_proj = nn.Linear(d_model, d_model)
        # The key projection is task-agnostic, so the model hoists it out and
        # passes pre-projected keys. own_keys=True keeps the predictor usable
        # standalone (tests, ablations) without leaving 9 dead copies of a
        # d_model x d_model Linear in the parameter tree when it does not.
        self.k_proj = nn.Linear(d_model, d_model) if own_keys else None
        self._scale = tau * d_model**0.5

        # --- Stage 2 ---
        canon_mean = torch.tensor(prior.canonical_mean, dtype=torch.float32)  # (K,2)
        canon_basis = torch.tensor(prior.canonical_basis, dtype=torch.float32)  # (Mc,2K)
        self.register_buffer("canon_mean", canon_mean)
        self.register_buffer("canon_basis", canon_basis)
        self.alpha_head = nn.Linear(d_model, canon_basis.shape[0])
        # Zero bias → population-average alpha ≈ 0 → start at the mean shape.
        # Weights keep their default init so different instance queries produce
        # different alphas; zeroing both would lock every slot together.
        nn.init.zeros_(self.alpha_head.bias)

        # --- Stage 3 ---
        # Measurement partner per landmark, from the clinical parameter each one
        # belongs to (PARAMS). The boundary being searched for is perpendicular
        # to the segment, so the segment direction is the direction to scan.
        _p = partner if partner is not None else [-1] * n_keypoints
        self.register_buffer("partner", torch.tensor(_p, dtype=torch.long))
        _d_fine = d_fine if d_fine is not None else d_model
        self.local_proj = nn.Linear(_d_fine, d_local)
        self.offset_head = nn.Sequential(
            nn.Linear(d_model + d_local * n_profile, d_model),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            # 2 outputs per bin: displacement along the measurement axis and
            # perpendicular to it. The frame matters — "slide along the wall"
            # and "move onto the wall" have different tolerances, and only the
            # second is what the endpoint error is actually about.
            nn.Linear(d_model, 2 * n_offset_bins),
        )

    def _axis_frame(self, pts: Tensor) -> tuple[Tensor, Tensor]:
        """Unit vectors along / perpendicular to each landmark's measurement axis.

        pts: (B,N,K,2) → two (B,N,K,2). Landmarks with no partner (partner<0)
        fall back to the image x-axis so the profile is still well-defined.
        """
        _idx = self.partner.clamp(min=0)
        _tgt = pts.index_select(-2, _idx)
        _u = _tgt - pts
        _has = (self.partner >= 0).view(1, 1, -1, 1)
        _u = torch.where(_has, _u, torch.zeros_like(_u))
        _n = _u.norm(dim=-1, keepdim=True)
        _u = torch.where(_n > 1e-6, _u / _n.clamp(min=1e-6), pts.new_tensor([1.0, 0.0]))
        _perp = torch.stack([-_u[..., 1], _u[..., 0]], dim=-1)
        return _u, _perp

    def _sample_profile(self, fine: Tensor, pts: Tensor, u: Tensor, half: Tensor) -> Tensor:
        """1D feature profile along u, centred on pts. → (B,N,K,n_profile*d_local)."""
        B, N, K, _ = pts.shape
        P = self._n_prof
        _t = torch.linspace(-1.0, 1.0, P, device=pts.device, dtype=torch.float32)
        # half is per-axis (normalized units); project it onto the scan direction
        # so the profile spans the same physical distance regardless of angle.
        _step = (u * half).norm(dim=-1, keepdim=True).unsqueeze(-2)  # (B,N,K,1,1)
        _loc = pts.unsqueeze(-2) + _t.view(1, 1, 1, P, 1) * _step * u.unsqueeze(-2)

        _grid = (2.0 * _loc.reshape(B, N * K, P, 2) - 1.0).clamp(-1.0, 1.0)
        _s = F.grid_sample(fine, _grid, mode="bilinear", align_corners=False)
        _s = _s.permute(0, 2, 3, 1)  # (B, N*K, P, C_fine)
        return self.local_proj(_s).reshape(B, N, K, P * self.local_proj.out_features)

    def forward(
        self,
        inst_feat: Tensor,
        lm_feat: Tensor,
        memory: Tensor | None = None,
        spatial_shape: tuple[int, int] | None = None,
        memory_pos: Tensor | None = None,
        mem_k: Tensor | None = None,
        fine: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        if memory is None or spatial_shape is None:
            raise ValueError("GeoSimCCPredictor requires memory and spatial_shape")
        B, N, _ = inst_feat.shape
        K = self._K
        hp, wp = spatial_shape

        # ── Stage 1: geometric coarse ────────────────────────────────────
        # Keys come pre-projected from the model: projecting memory is shared
        # work, and doing it per-task meant the same (B,S,D) tensor went through
        # nine separate Linears. Falls back to local projection so the predictor
        # stays usable standalone (tests, ablations).
        if mem_k is None:
            if self.k_proj is None:
                raise ValueError(
                    "GeoSimCCPredictor was built with own_keys=False and needs mem_k"
                )
            _m = memory + memory_pos if memory_pos is not None else memory
            mem_k = self.k_proj(_m)
        q = self.q_proj(lm_feat)  # (B,N,K,D)
        logits = torch.einsum("bnkd,bsd->bnks", q, mem_k) / self._scale
        heat = logits.float().softmax(dim=-1)  # fp32: soft-argmax is bf16-fragile

        xs, ys = normalized_grid(hp, wp, memory.device)
        coarse = torch.stack([(heat * xs).sum(-1), (heat * ys).sum(-1)], dim=-1)

        # ── Stage 2: shape projection ────────────────────────────────────
        alpha = self.alpha_head(inst_feat)  # (B,N,Mc)
        shape_c = (self.canon_mean.flatten() + alpha @ self.canon_basis).view(B, N, K, 2)
        # Uniform weights: confidence weighting needs a calibration check first,
        # and a sharp-but-wrong peak would otherwise get maximum authority.
        pts_shape = weighted_similarity_fit(shape_c, coarse.to(shape_c.dtype))
        pts = (1.0 - self._shape_w) * coarse.to(pts_shape.dtype) + self._shape_w * pts_shape

        # ── Stage 3: bounded offset from a 1D profile along the measurement axis ──
        _base = pts.detach().clamp(0.0, 1.0)
        _u, _perp = self._axis_frame(_base)
        # Window is in stage-1 grid cells: that is the accuracy stage 1 is asked
        # for, so it is also how far stage 3 must be able to reach.
        half = pts.new_tensor([self._window_cells / wp, self._window_cells / hp]).float()
        # Sampling `fine` when available is the point of this stage; memory at
        # ~50 original px per cell cannot resolve the boundary being sought.
        _src = fine if fine is not None else memory.transpose(1, 2).reshape(B, -1, hp, wp)
        local = self._sample_profile(_src, _base, _u, half)

        off_logits = self.offset_head(torch.cat([lm_feat, local], dim=-1))
        off_logits = off_logits.float().view(B, N, K, 2, self._n_bins)

        _bins = torch.arange(self._n_bins, device=lm_feat.device, dtype=torch.float32) + 0.5
        _bins = (_bins / self._n_bins) * 2.0 - 1.0  # (n_bins,) in [-1,1]
        _d = (off_logits.softmax(-1) * _bins).sum(-1)  # (B,N,K,2): along, perp
        _reach = (_u * half).norm(dim=-1, keepdim=True).float()
        delta = _d[..., :1] * _u.float() * _reach + _d[..., 1:] * _perp.float() * _reach

        coords = (pts.float() + delta).clamp(0.0, 1.0).to(lm_feat.dtype)
        return coords, heat


def build_coord_predictor(
    config: GeoSimCCCoordConfig,
    d_model: int,
    n_keypoints: int,
    task_shape_prior: TaskShapePrior | None = None,
    dropout: float = 0.0,
    task_id: str | None = None,
    d_fine: int | None = None,
    own_keys: bool = True,
) -> GeoSimCCPredictor:
    """Factory: create the coord predictor from config."""
    if task_shape_prior is None:
        raise ValueError(
            "GeoSimCCCoordConfig requires task_shape_prior — was shape_prior.json loaded?"
        )
    return GeoSimCCPredictor(
        d_model=d_model,
        n_keypoints=n_keypoints,
        prior=task_shape_prior,
        tau=config.tau,
        window_cells=config.window_cells,
        n_offset_bins=config.n_offset_bins,
        n_profile=config.n_profile,
        shape_weight=config.shape_weight,
        d_local=config.d_local,
        dropout=dropout,
        partner=measurement_partners(task_id, n_keypoints) if task_id else None,
        d_fine=d_fine,
        own_keys=own_keys,
    )
