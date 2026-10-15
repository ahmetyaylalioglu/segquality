"""3D VAR-style tokenizer: multi-scale **residual** VQ over a spatial grid.

Why this exists (advisor direction, 2026-10)
---------------------------------------------
The advisor asked us to bring the VAR (Visual Autoregressive Modeling,
NeurIPS 2024) *tokenizer* into our pipeline. VAR's Stage-1 is a VQ-VAE whose
quantizer is **multi-scale and residual**: instead of snapping one bottleneck
grid to the codebook once (what :class:`models.vae3d_vq.ConvVQVAE3D` does), it
encodes the feature map coarse-to-fine. At each scale ``pn`` it downsamples the
*residual* to ``pn^3``, snaps it to the shared codebook, upsamples the chosen
codes back to full resolution, applies a small conv correction ``Phi``, adds it
to ``f_hat`` and subtracts it from the running residual. Summing the scales
reconstructs the feature map coarse-to-fine, which is exactly the reconstruction
behaviour the advisor wants to test.

This is a faithful 3D port of ``_reference_repos/VAR/models/quant.py``
(``VectorQuantizer2``). The only substantive changes are dimensional:
``Conv2d -> Conv3d``; ``size=(pn, pn) -> (pn, pn, pn)``; and the upsample mode
``'bicubic' -> 'trilinear'`` (bicubic is 2D-only). Distributed all-reduce and
the EMA usage buffers of the original are dropped (single-GPU training here);
a lightweight code-usage / perplexity diagnostic is kept for collapse checks.

The encoder/decoder are intentionally **identical** to ``ConvVQVAE3D`` so that a
QC comparison isolates the quantizer: single-scale VQ vs multi-scale residual VQ.
Nothing here imports or edits the protected ``models/vae3d_vq.py``.

Public API matches the other 3D models: ``forward`` returns
``(logits, vq_loss, info)``; ``reconstruct``/``encode``/``decode`` behave
analogously, so the existing 3D correlation/QC scripts can score this model.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

# Same channel schedule as ConvVQVAE3D: 4 stride-2 stages, 64->32->16->8->4.
_CHANNEL_MULTS: tuple[int, ...] = (1, 2, 4, 8)


class Phi3D(nn.Conv3d):
    """Per-scale residual correction conv (VAR ``Phi``, ported to 3D).

    Mixes a fraction ``resi_ratio`` of a 3x3x3 conv into the upsampled code map::

        out = h * (1 - resi_ratio) + conv(h) * resi_ratio

    With ``quant_resi == 0`` the caller substitutes ``nn.Identity`` instead.
    """

    def __init__(self, embed_dim: int, quant_resi: float) -> None:
        ks = 3
        super().__init__(embed_dim, embed_dim, kernel_size=ks, stride=1, padding=ks // 2)
        self.resi_ratio = abs(quant_resi)

    def forward(self, h: Tensor) -> Tensor:  # type: ignore[override]
        return h.mul(1 - self.resi_ratio) + super().forward(h).mul(self.resi_ratio)


class MultiScaleVQ3D(nn.Module):
    """Multi-scale residual vector quantizer for 5D ``(B, C, D, H, W)`` grids.

    Port of VAR ``VectorQuantizer2.forward`` to 3D. ``v_patch_nums`` lists the
    cube side lengths from small to large; the last entry MUST equal the
    bottleneck side length (so the finest scale is lossless-resolution).
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        v_patch_nums: tuple[int, ...],
        using_znorm: bool = False,
        beta: float = 0.25,
        quant_resi: float = 0.5,
    ) -> None:
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.Cvae = int(embedding_dim)
        self.v_patch_nums = tuple(int(p) for p in v_patch_nums)
        self.using_znorm = bool(using_znorm)
        self.beta = float(beta)

        self.embedding = nn.Embedding(self.vocab_size, self.Cvae)
        bound = 1.0 / self.vocab_size
        self.embedding.weight.data.uniform_(-bound, bound)

        # One residual-correction conv per scale (non-shared: K is small for 3D).
        SN = len(self.v_patch_nums)
        if abs(quant_resi) > 1e-6:
            self.quant_resi = nn.ModuleList([Phi3D(self.Cvae, quant_resi) for _ in range(SN)])
        else:
            self.quant_resi = nn.ModuleList([nn.Identity() for _ in range(SN)])

    def _nearest_idx(self, rest_NC: Tensor) -> Tensor:
        """Nearest codebook index for each row of ``rest_NC`` (N, C)."""
        if self.using_znorm:
            z = F.normalize(rest_NC, dim=-1)
            w = F.normalize(self.embedding.weight.data, dim=1)
            return torch.argmax(z @ w.t(), dim=1)
        w = self.embedding.weight.data
        d = (
            rest_NC.square().sum(1, keepdim=True)
            + w.square().sum(1)
            - 2.0 * rest_NC @ w.t()
        )
        return torch.argmin(d, dim=1)

    def forward(self, f: Tensor) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
        """Quantize ``f`` (B, C, D, H, W). Returns ``(f_hat, vq_loss, info)``.

        ``f_hat`` matches the input shape and carries a straight-through gradient
        to the encoder. ``vq_loss`` is the VAR multi-scale objective averaged over
        scales; ``info`` reports ``codebook_used`` and ``perplexity``.
        """
        B, C, D, H, W = f.shape
        f = f.float()
        f_no_grad = f.detach()
        f_rest = f_no_grad.clone()
        f_hat = torch.zeros_like(f_rest)

        SN = len(self.v_patch_nums)
        vocab_hit = torch.zeros(self.vocab_size, device=f.device)
        vq_loss: Tensor = f.new_zeros(())

        for si, pn in enumerate(self.v_patch_nums):  # small -> large
            last = si == SN - 1
            if last:
                rest = f_rest
            else:
                rest = F.interpolate(f_rest, size=(pn, pn, pn), mode="area")
            rest_NC = rest.permute(0, 2, 3, 4, 1).reshape(-1, C)
            idx_N = self._nearest_idx(rest_NC)

            vocab_hit += idx_N.bincount(minlength=self.vocab_size).float()

            idx_grid = idx_N.view(B, pn, pn, pn)
            codes = self.embedding(idx_grid).permute(0, 4, 1, 2, 3).contiguous()
            if not last:
                codes = F.interpolate(codes, size=(D, H, W), mode="trilinear", align_corners=False)
            h = self.quant_resi[si](codes)
            f_hat = f_hat + h
            f_rest = f_rest - h

            # VAR loss: commitment (pull encoder to f_hat) + codebook (pull codes to f).
            vq_loss = vq_loss + F.mse_loss(f_hat.data, f) * self.beta + F.mse_loss(f_hat, f_no_grad)

        vq_loss = vq_loss / SN
        # Straight-through: value == f_hat, gradient flows to the encoder.
        f_hat = (f_hat.data - f_no_grad) + f

        avg_probs = vocab_hit / vocab_hit.sum().clamp_min(1.0)
        perplexity = torch.exp(-(avg_probs * (avg_probs + 1e-10).log()).sum())
        info = {"perplexity": perplexity, "codebook_used": (avg_probs > 0).sum()}
        return f_hat, vq_loss, info


class VARVQVAE3D(nn.Module):
    """VAR-style 3D VQ-VAE: same conv encoder/decoder as ``ConvVQVAE3D`` but with
    a multi-scale residual quantizer at the bottleneck."""

    def __init__(
        self,
        base_channels: int = 16,
        num_embeddings: int = 512,
        embedding_dim: int = 64,
        input_size: tuple[int, int, int] = (64, 64, 64),
        in_channels: int = 1,
        v_patch_nums: tuple[int, ...] | None = None,
        using_znorm: bool = False,
        beta: float = 0.25,
        quant_resi: float = 0.5,
    ) -> None:
        super().__init__()
        self.base_channels = int(base_channels)
        self.num_embeddings = int(num_embeddings)
        self.embedding_dim = int(embedding_dim)
        self.input_size = (int(input_size[0]), int(input_size[1]), int(input_size[2]))
        self.in_channels = int(in_channels)

        n_stages = len(_CHANNEL_MULTS)
        divisor = 2**n_stages
        if any(s % divisor for s in self.input_size):
            raise ValueError(
                f"input_size {self.input_size} must be divisible by {divisor} "
                f"for {n_stages} stride-2 stages"
            )
        self._feat_dhw = tuple(s // divisor for s in self.input_size)  # e.g. (4,4,4)
        feat = self._feat_dhw[0]
        if not (self._feat_dhw[0] == self._feat_dhw[1] == self._feat_dhw[2]):
            raise ValueError(f"multi-scale VQ needs a cubic bottleneck, got {self._feat_dhw}")

        # Default scale schedule up to the bottleneck side (e.g. 4 -> (1,2,3,4)).
        if v_patch_nums is None:
            v_patch_nums = tuple(range(1, feat + 1))
        if v_patch_nums[-1] != feat:
            raise ValueError(
                f"v_patch_nums[-1] ({v_patch_nums[-1]}) must equal bottleneck side ({feat})"
            )
        self.v_patch_nums = tuple(int(p) for p in v_patch_nums)

        channels = [self.base_channels * m for m in _CHANNEL_MULTS]
        self._channels = channels
        self._bottleneck_channels = channels[-1]

        self.enc_stages = self._build_encoder(channels)
        self.pre_quant = nn.Conv3d(self._bottleneck_channels, self.embedding_dim, kernel_size=1)
        self.quantizer = MultiScaleVQ3D(
            self.num_embeddings, self.embedding_dim, self.v_patch_nums,
            using_znorm=using_znorm, beta=beta, quant_resi=quant_resi,
        )
        self.post_quant = nn.Conv3d(self.embedding_dim, self._bottleneck_channels, kernel_size=1)
        self.dec_stages = self._build_decoder(channels)

    # -- construction (identical to ConvVQVAE3D, no skip) -------------------
    def _build_encoder(self, channels: list[int]) -> nn.ModuleList:
        stages: list[nn.Module] = []
        prev = self.in_channels
        for ch in channels:
            stages.append(
                nn.Sequential(
                    nn.Conv3d(prev, ch, kernel_size=4, stride=2, padding=1),
                    nn.BatchNorm3d(ch),
                    nn.LeakyReLU(0.2, inplace=True),
                )
            )
            prev = ch
        return nn.ModuleList(stages)

    def _build_decoder(self, channels: list[int]) -> nn.ModuleList:
        rev = list(reversed(channels))
        out_channels = rev[1:] + [self.in_channels]
        stages: list[nn.Module] = []
        for i, (prev, out) in enumerate(zip(rev, out_channels)):
            is_last = i == len(rev) - 1
            layers: list[nn.Module] = [
                nn.ConvTranspose3d(prev, out, kernel_size=4, stride=2, padding=1)
            ]
            if not is_last:
                layers += [nn.BatchNorm3d(out), nn.LeakyReLU(0.2, inplace=True)]
            stages.append(nn.Sequential(*layers))
        return nn.ModuleList(stages)

    # -- core API -----------------------------------------------------------
    def _encode_features(self, x: Tensor) -> Tensor:
        h = x
        for stage in self.enc_stages:
            h = stage(h)
        return self.pre_quant(h)

    def _decode_features(self, f_hat: Tensor) -> Tensor:
        h = self.post_quant(f_hat)
        for stage in self.dec_stages:
            h = stage(h)
        return h

    def encode(self, x: Tensor) -> Tensor:
        f_hat, _, _ = self.quantizer(self._encode_features(x))
        return torch.flatten(f_hat, start_dim=1)

    def decode(self, f_hat: Tensor) -> Tensor:
        return self._decode_features(f_hat)

    def quantize(self, x: Tensor) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
        return self.quantizer(self._encode_features(x))

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
        f_hat, vq_loss, info = self.quantizer(self._encode_features(x))
        logits = self._decode_features(f_hat)
        return logits, vq_loss, info

    def reconstruct(self, x: Tensor) -> Tensor:
        logits, _, _ = self.forward(x)
        return torch.sigmoid(logits)

    # -- introspection ------------------------------------------------------
    @property
    def feature_shape(self) -> tuple[int, int, int]:
        return self._feat_dhw

    @property
    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
