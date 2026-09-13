"""3D VQ-VAE conditioned on geometric mask features (WP-2).

Motivation
----------
The plain :class:`models.vae3d_vq.ConvVQVAE3D` scores a mask purely by how well its
learned shape codebook reconstructs it. WP-2 asks a narrow question: if we ALSO
hand the decoder a few geometric descriptors of the input mask (topology + squashed
shape, see :mod:`quality.geom_features`), does the GT-free QC correlation improve?

Design constraints (why this is a NEW file, not an edit)
--------------------------------------------------------
* ``models/vae3d_vq.py`` is untouched — this class *subclasses* ``ConvVQVAE3D`` and
  reuses its encoder, quantizer and decoder verbatim.
* ``cond_dim=0`` is a hard no-op: no conditioning modules are built and ``forward``
  delegates to the parent, so behaviour AND ``param_count`` are byte-identical to the
  base model. Conditioning only exists when ``cond_dim > 0``.
* **Features are computed INSIDE ``forward`` from the input ``x``.** Downstream
  scoring (``quality.score`` / ``evaluate_correlation_3d_vq``) only ever calls
  ``model.reconstruct(x)`` / ``model.forward(x)`` with the mask alone — so a model
  that derives its own conditioning from ``x`` is scored by the existing evaluator
  with **zero changes**.

Where conditioning is injected
------------------------------
At the **decoder input** (the ``post_quant`` output, the 4^3 bottleneck): the
``cond_dim``-wide feature embedding is broadcast across the 4^3 grid and concatenated
onto the first decoder stage's input. This keeps the encoder, ``pre_quant``,
quantizer and codebook completely unchanged — so perplexity / usage stay directly
comparable to the WP-1 sweep, and the feature can only *complement* the discrete
latent, never replace or bypass it.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn

from models.vae3d_vq import ConvVQVAE3D
from quality.geom_features import compute_features_batch


class ConvVQVAE3D_cond(ConvVQVAE3D):
    """``ConvVQVAE3D`` with optional geometric-feature conditioning of the decoder.

    Extra args over the base model:

    * ``cond_dim`` — width of the feature embedding concatenated at the decoder
      input. ``0`` disables conditioning entirely (identical to the base model).
      Kept small (default project value 16) so it cannot dominate the 64-channel
      quantized latent.
    * ``feature_names`` — ordered list of geometric features to feed (see
      :func:`quality.geom_features.resolve_feature_names`). Required when
      ``cond_dim > 0``. Its length fixes the conditioning MLP's input width and the
      standardisation buffers, so it is part of the checkpoint contract.
    * ``cond_hidden`` — hidden width of the 2-layer conditioning MLP.
    """

    def __init__(
        self,
        base_channels: int = 16,
        num_embeddings: int = 512,
        embedding_dim: int = 64,
        commitment_cost: float = 0.25,
        input_size: tuple[int, int, int] = (64, 64, 64),
        in_channels: int = 1,
        use_skip: bool = False,
        cond_dim: int = 0,
        feature_names: list[str] | None = None,
        cond_hidden: int = 32,
    ) -> None:
        super().__init__(
            base_channels=base_channels,
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            commitment_cost=commitment_cost,
            input_size=input_size,
            in_channels=in_channels,
            use_skip=use_skip,
        )
        self.cond_dim = int(cond_dim)
        self.feature_names = list(feature_names) if feature_names else []
        self.cond_hidden = int(cond_hidden)

        if self.cond_dim > 0:
            if not self.feature_names:
                raise ValueError("cond_dim > 0 requires a non-empty feature_names list")
            n_feat = len(self.feature_names)
            # 2-layer MLP: raw feature vector -> cond_dim embedding.
            self.cond_mlp = nn.Sequential(
                nn.Linear(n_feat, self.cond_hidden),
                nn.ReLU(inplace=True),
                nn.Linear(self.cond_hidden, self.cond_dim),
            )
            # Standardisation buffers (identity by default). Training fills these
            # with the train-split feature mean/std so the MLP sees ~unit-scale
            # inputs; they travel with the checkpoint so eval matches training.
            self.register_buffer("feat_mean", torch.zeros(n_feat))
            self.register_buffer("feat_std", torch.ones(n_feat))
            # The first decoder stage now also ingests the broadcast cond map, so
            # its input channel count grows by cond_dim. Rebuild only that stage;
            # every other stage (and its weights) is left exactly as the parent
            # built it.
            self._grow_first_decoder_stage(self.cond_dim)

    # -- construction -------------------------------------------------------
    def _grow_first_decoder_stage(self, extra: int) -> None:
        """Replace decoder stage 0's transpose-conv with one whose input has
        ``extra`` more channels (for the concatenated conditioning map), keeping
        its output channels and the following BatchNorm/activation intact."""
        old_stage = self.dec_stages[0]
        old_conv = old_stage[0]  # ConvTranspose3d
        new_conv = nn.ConvTranspose3d(
            old_conv.in_channels + extra,
            old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
        )
        self.dec_stages[0] = nn.Sequential(new_conv, *list(old_stage[1:]))

    # -- conditioning -------------------------------------------------------
    def set_feature_stats(self, mean: Tensor, std: Tensor) -> None:
        """Set the standardisation buffers from training-split statistics.

        ``std`` is floored at a small epsilon so a constant feature (zero variance
        across the train split) does not produce NaNs.
        """
        mean = torch.as_tensor(mean, dtype=self.feat_mean.dtype)
        std = torch.as_tensor(std, dtype=self.feat_std.dtype).clamp_min(1e-6)
        if mean.shape != self.feat_mean.shape or std.shape != self.feat_std.shape:
            raise ValueError(
                f"expected stats of shape {tuple(self.feat_mean.shape)}, "
                f"got mean {tuple(mean.shape)} / std {tuple(std.shape)}"
            )
        self.feat_mean.copy_(mean)
        self.feat_std.copy_(std)

    def _cond_vector(self, x: Tensor) -> Tensor:
        """Geometric features of ``x`` -> standardised -> MLP -> ``(B, cond_dim)``."""
        feats = compute_features_batch(x, self.feature_names, device=x.device)
        feats = (feats - self.feat_mean) / self.feat_std
        return self.cond_mlp(feats)

    def _decode_features_cond(self, z_q: Tensor, feats: list[Tensor], x: Tensor) -> Tensor:
        """Decode as the parent does, but concatenate the broadcast conditioning
        map onto the first decoder stage's input."""
        h = self.post_quant(z_q)
        cond = self._cond_vector(x)                       # (B, cond_dim)
        b, _, d, hh, w = h.shape
        cond_map = cond.view(b, self.cond_dim, 1, 1, 1).expand(b, self.cond_dim, d, hh, w)
        n = len(self.dec_stages)
        for i, stage in enumerate(self.dec_stages):
            if self.use_skip:                             # skip concat as in the parent
                h = torch.cat([h, feats[n - 1 - i]], dim=1)
            if i == 0:                                    # conditioning at decoder input
                h = torch.cat([h, cond_map], dim=1)
            h = stage(h)
        return h

    # -- API ----------------------------------------------------------------
    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
        if self.cond_dim == 0:
            # No-op path: identical to the base model (and identical param_count).
            return super().forward(x)
        z_e, feats = self._encode_features(x)
        z_q, vq_loss, info = self.quantizer(z_e)
        logits = self._decode_features_cond(z_q, feats, x)
        return logits, vq_loss, info
