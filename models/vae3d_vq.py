"""3D VQ-VAE with a **spatial** discrete latent grid (Phase 9).

Why this exists (mean-reversion cure)
-------------------------------------
The continuous-latent :class:`models.vae3d.ConvVAE3D` (untouched) suffers
mean-reversion: its single global latent vector, regularised by KL, drifts toward
the dataset mean and the decoder emits one generic liver for everyone
(cross-reconstruction ``ratio`` ~= 1.0, recon-pairwise Dice 0.99 vs input 0.73).

This model attacks the root cause two ways the advisor recommended:

1. **Vector quantization over a spatial grid.** Instead of one global vector, the
   encoder's ``(B, C, 4, 4, 4)`` bottleneck is projected to ``embedding_dim`` and
   every one of the 4x4x4 = 64 spatial positions is snapped to its nearest entry in
   a shared **codebook** (``num_embeddings`` discrete vectors). A discrete,
   position-local code cannot be "averaged away" by KL, so the decoder is forced to
   respect where the input actually differs — directly countering mean-reversion.
2. **Optional U-Net skip connections** (``use_skip``). When enabled, each encoder
   stage's feature map is concatenated into the matching-resolution decoder stage,
   pushing fine encoder detail past the bottleneck. Off by default: we first test
   whether quantization *alone* fixes mean-reversion, then add skips if needed.

Same public API as ``ConvVAE3D`` where it matters for downstream code:
``reconstruct``/``encode``/``decode`` behave analogously, so the existing 3D
correlation/diagnosis scripts can score this model. ``forward`` returns
``(logits, vq_loss, info)`` instead of ``(logits, mu, logvar)`` — the VQ objective
has no KL term — so it gets its own training loop in ``train_vae3d_vq.py``.

Nothing here imports or modifies ``models/vae3d.py``; the two live side by side.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

# Channel multipliers over ``base_channels``; length = number of stride-2 stages.
# Four stages take 64 -> 32 -> 16 -> 8 -> 4 (identical schedule to ConvVAE3D).
_CHANNEL_MULTS: tuple[int, ...] = (1, 2, 4, 8)


class VectorQuantizer3D(nn.Module):
    """Snap each spatial position of a feature map to the nearest codebook vector.

    Implements the standard VQ-VAE (van den Oord et al., 2017) quantizer with the
    straight-through estimator, generalised to a 5D ``(B, D, d, h, w)`` input. The
    codebook loss pulls the used embeddings toward the encoder outputs, while the
    commitment loss (weighted by ``commitment_cost``) pulls the encoder outputs
    toward their chosen embeddings so the encoder "commits" to codes.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, commitment_cost: float,
                 vq_mode: str = "standard") -> None:
        super().__init__()
        self.num_embeddings = int(num_embeddings)
        self.embedding_dim = int(embedding_dim)
        self.commitment_cost = float(commitment_cost)
        # ``vq_mode`` selects the VQ objective (default = standard van den Oord VQ-VAE):
        #   "standard"                -> vq_loss = codebook_loss + commitment_cost * commitment_loss
        #   "commitment_only"         -> vq_loss = commitment_cost * commitment_loss (NO codebook_loss)
        #   "commitment_only_scaled"  -> commitment_only, BUT codes init at N(0,1) scale
        #   "commitment_only_cosine"  -> commitment_only, BUT encoder + codes L2-normalised (cosine VQ)
        #
        # Advisor rationale for dropping codebook_loss: here the VQ-VAE is used for QC/denoising.
        # ``codebook_loss = MSE(z_q, z.detach())`` is the ONLY gradient path to the codebook, so
        # dropping it (with no EMA) FREEZES the codebook and forces the encoder to align to fixed
        # reference codes -- we do NOT want degraded-mask features to drag the codebook toward
        # themselves; we want a clean reference recovered from them.
        #
        # Why the *_scaled / *_cosine variants: plain "commitment_only" freezes codes at the tiny
        # uniform(-1/K,1/K) init (~+-0.002) while the encoder output is free to grow (measured std
        # ~19), so commitment = ||z - z_q||^2 diverges (0.7 -> 964). Two fixes that KEEP the codebook
        # frozen (honouring the rationale): (scaled) init codes at N(0,1) so a frozen codebook already
        # lives at feature scale; (cosine) L2-normalise the encoder AND codes so magnitude cannot
        # diverge at all. EMA is deliberately NOT offered: EMA would move the codes toward the
        # encoder features -- exactly what the advisor wants to avoid -- so it is the wrong cure here.
        _MODES = ("standard", "commitment_only", "commitment_only_scaled", "commitment_only_cosine")
        if vq_mode not in _MODES:
            raise ValueError(f"vq_mode must be one of {_MODES}, got {vq_mode!r}")
        self.vq_mode = vq_mode
        self._frozen = vq_mode.startswith("commitment_only")   # no codebook_loss
        self._cosine = vq_mode == "commitment_only_cosine"      # L2-normalise enc + codes
        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)
        if vq_mode in ("commitment_only_scaled", "commitment_only_cosine"):
            # Unit-scale init so a FROZEN codebook sits at the encoder-feature scale.
            self.embedding.weight.data.normal_(0.0, 1.0)
        else:
            # Uniform init in [-1/K, 1/K] is the usual VQ-VAE choice: small, unbiased.
            bound = 1.0 / self.num_embeddings
            self.embedding.weight.data.uniform_(-bound, bound)

    def forward(self, z_e: Tensor) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
        """Quantize ``z_e`` (B, D, d, h, w). Returns ``(z_q, vq_loss, info)``.

        ``z_q`` has the same shape as ``z_e`` and carries a straight-through
        gradient back to the encoder. ``info`` holds ``perplexity`` and
        ``codebook_used`` (both diagnostics for codebook collapse).
        """
        # (B, D, d, h, w) -> (B, d, h, w, D) -> (N, D) so each row is one position.
        z = z_e.permute(0, 2, 3, 4, 1).contiguous()
        # ``zc``/``wc`` are the vectors actually quantized/compared. For cosine VQ both the
        # encoder positions and the codes are L2-normalised (bounds the magnitude so commitment
        # cannot diverge); otherwise they are the raw encoder output and codebook (identical to
        # the standard path -- default behaviour is byte-for-byte unchanged).
        zc = F.normalize(z, dim=-1) if self._cosine else z
        wc = F.normalize(self.embedding.weight, dim=1) if self._cosine else self.embedding.weight
        flat = zc.reshape(-1, self.embedding_dim)

        # Squared L2 distance to every codebook vector, then nearest index.
        dist = (
            flat.pow(2).sum(1, keepdim=True)
            - 2.0 * flat @ wc.t()
            + wc.pow(2).sum(1)
        )
        indices = dist.argmin(1)                    # (N,)
        z_q_flat = wc[indices]                       # (N, D) — exact codebook rows
        z_q = z_q_flat.view(zc.shape)                # (B, d, h, w, D)

        # VQ objective: codebook (move embeddings) + commitment (move encoder).
        commitment_loss = F.mse_loss(zc, z_q.detach())
        if self._frozen:
            # Codebook loss dropped -> embedding.weight receives NO gradient -> codebook is
            # frozen at init; only the encoder is trained (to commit to fixed reference codes).
            vq_loss = self.commitment_cost * commitment_loss
        else:
            codebook_loss = F.mse_loss(z_q, zc.detach())
            vq_loss = codebook_loss + self.commitment_cost * commitment_loss

        # Straight-through: value equals z_q, but gradient flows to z_e unchanged (through the
        # L2-normalise in the cosine case, which is differentiable).
        z_q = zc + (z_q - zc).detach()
        z_q = z_q.permute(0, 4, 1, 2, 3).contiguous()  # back to (B, D, d, h, w)

        # Diagnostics: how many codes are actually used, and perplexity (effective
        # number of codes). Low perplexity => codebook collapse => diversity lost.
        one_hot = F.one_hot(indices, self.num_embeddings).float()
        avg_probs = one_hot.mean(0)
        perplexity = torch.exp(-(avg_probs * (avg_probs + 1e-10).log()).sum())
        codebook_used = (avg_probs > 0).sum()
        info = {"perplexity": perplexity, "codebook_used": codebook_used}
        return z_q, vq_loss, info


class ConvVQVAE3D(nn.Module):
    """Convolutional VQ-VAE for single-channel binary volumes (spatial codebook)."""

    def __init__(
        self,
        base_channels: int = 16,
        num_embeddings: int = 512,
        embedding_dim: int = 64,
        commitment_cost: float = 0.25,
        input_size: tuple[int, int, int] = (64, 64, 64),
        in_channels: int = 1,
        use_skip: bool = False,
        vq_mode: str = "standard",
    ) -> None:
        super().__init__()
        self.base_channels = int(base_channels)
        self.num_embeddings = int(num_embeddings)
        self.embedding_dim = int(embedding_dim)
        self.commitment_cost = float(commitment_cost)
        self.input_size = (int(input_size[0]), int(input_size[1]), int(input_size[2]))
        self.in_channels = int(in_channels)
        self.use_skip = bool(use_skip)
        self.vq_mode = str(vq_mode)

        n_stages = len(_CHANNEL_MULTS)
        divisor = 2**n_stages
        if any(s % divisor for s in self.input_size):
            raise ValueError(
                f"input_size {self.input_size} must be divisible by {divisor} "
                f"for {n_stages} stride-2 stages"
            )
        self._feat_dhw = tuple(s // divisor for s in self.input_size)  # e.g. (4,4,4)
        channels = [self.base_channels * m for m in _CHANNEL_MULTS]    # [16,32,64,128]
        self._channels = channels
        self._bottleneck_channels = channels[-1]

        # Encoder as separate stages so skip connections can tap each feature map.
        self.enc_stages = self._build_encoder(channels)
        # Project bottleneck channels -> embedding_dim (the quantized grid), and back.
        self.pre_quant = nn.Conv3d(self._bottleneck_channels, self.embedding_dim, kernel_size=1)
        self.quantizer = VectorQuantizer3D(self.num_embeddings, self.embedding_dim,
                                           self.commitment_cost, vq_mode=self.vq_mode)
        self.post_quant = nn.Conv3d(self.embedding_dim, self._bottleneck_channels, kernel_size=1)
        self.dec_stages = self._build_decoder(channels)

    # -- construction helpers ----------------------------------------------
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
        """Mirror the encoder. With ``use_skip`` each stage input is doubled because
        the matching encoder feature map is concatenated in before the transpose."""
        rev = list(reversed(channels))                    # [128, 64, 32, 16]
        out_channels = rev[1:] + [self.in_channels]       # [64, 32, 16, in_channels]
        stages: list[nn.Module] = []
        for i, (prev, out) in enumerate(zip(rev, out_channels)):
            in_ch = prev * 2 if self.use_skip else prev    # skip concat doubles input
            is_last = i == len(rev) - 1
            layers: list[nn.Module] = [
                nn.ConvTranspose3d(in_ch, out, kernel_size=4, stride=2, padding=1)
            ]
            if not is_last:                                # final stage -> raw logits
                layers += [nn.BatchNorm3d(out), nn.LeakyReLU(0.2, inplace=True)]
            stages.append(nn.Sequential(*layers))
        return nn.ModuleList(stages)

    # -- core API -----------------------------------------------------------
    def _encode_features(self, x: Tensor) -> tuple[Tensor, list[Tensor]]:
        """Run the encoder, returning the pre-quant projection and stage feats."""
        feats: list[Tensor] = []
        h = x
        for stage in self.enc_stages:
            h = stage(h)
            feats.append(h)          # feats[0]=finest ... feats[-1]=bottleneck res
        z_e = self.pre_quant(h)
        return z_e, feats

    def _decode_features(self, z_q: Tensor, feats: list[Tensor]) -> Tensor:
        """Decode a quantized grid back to logits, optionally using skip feats."""
        h = self.post_quant(z_q)
        n = len(self.dec_stages)
        for i, stage in enumerate(self.dec_stages):
            if self.use_skip:
                # Stage i input is at the resolution of encoder feats[n-1-i].
                h = torch.cat([h, feats[n - 1 - i]], dim=1)
            h = stage(h)
        return h

    def encode(self, x: Tensor) -> Tensor:
        """Return the (straight-through) quantized latent, flattened per sample."""
        z_e, feats = self._encode_features(x)
        z_q, _, _ = self.quantizer(z_e)
        return torch.flatten(z_q, start_dim=1)

    def decode(self, z_q: Tensor) -> Tensor:
        """Map a quantized grid ``(B, embedding_dim, *feat)`` to logits.

        Skip connections need encoder features, which a bare latent lacks, so this
        convenience path requires ``use_skip=False`` (used by round-trip tests).
        """
        if self.use_skip:
            raise RuntimeError("decode(z) without encoder feats is unavailable when use_skip=True")
        return self._decode_features(z_q, feats=[])

    def quantize(self, x: Tensor) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
        """Encode and quantize; returns ``(z_q_grid, vq_loss, info)``."""
        z_e, _ = self._encode_features(x)
        return self.quantizer(z_e)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
        z_e, feats = self._encode_features(x)
        z_q, vq_loss, info = self.quantizer(z_e)
        logits = self._decode_features(z_q, feats)
        return logits, vq_loss, info

    def reconstruct(self, x: Tensor) -> Tensor:
        """Reconstruction probabilities (sigmoid applied) — same contract as 2D/3D VAE."""
        logits, _, _ = self.forward(x)
        return torch.sigmoid(logits)

    # -- introspection ------------------------------------------------------
    @property
    def feature_shape(self) -> tuple[int, int, int]:
        return self._feat_dhw

    @property
    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
