"""3D SimVQ-VAE — codebook-collapse cure via one shared linear layer (Phase 12).

Why this exists (advisor's paper)
---------------------------------
Our VQ ablation found **codebook collapse**: vanilla VQ used ~13/512 codes, yet
that collapse gave the *strongest* QC signal (``quant_error`` rho ~0.86). The
advisor sent "Addressing Representation Collapse in Vector Quantized Models with
One Linear Layer" (SimVQ, ICCV 2025), which explains the root cause and offers a
minimal cure. This model implements it so we can test the hypothesis directly and
compare against every prior 3D method under the byte-identical exam.

The SimVQ idea (root cause = *disjoint optimization*)
-----------------------------------------------------
Standard VQ only back-props into the **selected** nearest codes (the straight-
through estimator zeroes the rest), so unused codes never move and stay stranded
at their init — collapse. SimVQ reparametrizes the codebook ``C -> C @ W`` with a
**single** shared linear map ``W in R^{d x d}``:

* ``C`` (the base codebook) is a **frozen** random Gaussian matrix — never trained.
* ``W`` (one linear layer, no bias) is the **only** trained codebook parameter.

Because ``W`` is shared across *all* codes, one gradient step through any selected
code moves the ENTIRE effective codebook ``C @ W`` jointly. Unused codes ride along
and stay in the encoder's neighbourhood -> ~100% utilization instead of ~1-3%.
Vanilla VQ is the special case ``W = I``.

Objective (paper Algorithm 1), with ``q_k`` the frozen base code chosen for a
position and ``e_k = q_k @ W`` its effective (projected) code::

    k        = argmin_j || z_e - (q_j @ W) ||^2          # nearest in effective space
    L_commit = || z_e - sg(e_k) ||^2                     # pull encoder to code
    L_code   = || sg(z_e) - e_k ||^2                     # pull code (via W) to encoder
    vq_loss  = L_code + commitment_cost * L_commit
    z_q      = z_e + (e_k - z_e).detach()                # straight-through

Note ``L_code`` gradients reach ``W`` only (``C`` is frozen), so the effective
codebook still moves — the frozen base is just a fixed random anchor.

Nothing here imports-and-mutates ``models/vae3d_vq.py``; ``ConvVQVAE3D_SimVQ``
*subclasses* :class:`~models.vae3d_vq.ConvVQVAE3D` and only swaps the quantizer,
so encoder/decoder/skip/`reconstruct` behave identically and every downstream 3D
script can score it unchanged.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from models.vae3d_vq import ConvVQVAE3D


class SimVQuantizer3D(nn.Module):
    """SimVQ quantizer: frozen Gaussian base codebook + one learnable linear map.

    The effective codebook is ``C @ W`` where ``C`` (``self.codebook``) is a frozen
    ``N(0, 1)`` embedding and ``W`` (``self.proj``) is a single bias-free
    :class:`~torch.nn.Linear` shared across all codes. Drop-in compatible with
    :class:`~models.vae3d_vq.VectorQuantizer3D`: same ``forward`` signature, same
    ``perplexity``/``codebook_used`` diagnostics, same straight-through gradient.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int,
                 commitment_cost: float) -> None:
        super().__init__()
        self.num_embeddings = int(num_embeddings)
        self.embedding_dim = int(embedding_dim)
        self.commitment_cost = float(commitment_cost)
        # ``vq_mode`` kept for API parity with VectorQuantizer3D (report/logging).
        self.vq_mode = "simvq"

        # Base codebook C: random Gaussian, FROZEN (requires_grad=False). It is a
        # fixed anchor — the paper never trains it; all learning goes through W.
        self.codebook = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.codebook.weight.data.normal_(0.0, 1.0)
        self.codebook.weight.requires_grad_(False)

        # W: the single shared linear layer (no bias). Its default init
        # (uniform +-1/sqrt(d)) keeps the effective codebook C@W at ~unit scale, so
        # a fresh model already lives at the encoder-feature scale.
        self.proj = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)

    def effective_codebook(self) -> Tensor:
        """Return the effective codebook ``C @ W`` as ``(num_embeddings, D)``.

        This is the codebook the encoder is actually quantized against; downstream
        ``quant_error``-style scores must read THIS (not the frozen base ``C``).
        """
        return self.proj(self.codebook.weight)

    def forward(self, z_e: Tensor) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
        """Quantize ``z_e`` (B, D, d, h, w). Returns ``(z_q, vq_loss, info)``."""
        # (B, D, d, h, w) -> (B, d, h, w, D) -> (N, D): one row per spatial position.
        z = z_e.permute(0, 2, 3, 4, 1).contiguous()
        flat = z.reshape(-1, self.embedding_dim)

        # Effective codebook C@W — depends on W, so gradients through it train W.
        weight = self.effective_codebook()          # (K, D)

        # Squared L2 distance to every effective code, then nearest index.
        dist = (
            flat.pow(2).sum(1, keepdim=True)
            - 2.0 * flat @ weight.t()
            + weight.pow(2).sum(1)
        )
        indices = dist.argmin(1)                     # (N,)
        z_q_flat = weight[indices]                   # (N, D) — effective code rows
        z_q = z_q_flat.view(z.shape)                 # (B, d, h, w, D)

        # SimVQ objective: commitment moves the encoder; codebook loss moves W
        # (the frozen base C receives no gradient). Both terms always present.
        commitment_loss = F.mse_loss(z, z_q.detach())
        codebook_loss = F.mse_loss(z_q, z.detach())
        vq_loss = codebook_loss + self.commitment_cost * commitment_loss

        # Straight-through: value = z_q, gradient flows to z_e unchanged.
        z_q = z + (z_q - z).detach()
        z_q = z_q.permute(0, 4, 1, 2, 3).contiguous()  # back to (B, D, d, h, w)

        # Diagnostics: utilization + perplexity. SimVQ's whole point is that these
        # stay HIGH (near full codebook) where vanilla VQ collapses.
        one_hot = F.one_hot(indices, self.num_embeddings).float()
        avg_probs = one_hot.mean(0)
        perplexity = torch.exp(-(avg_probs * (avg_probs + 1e-10).log()).sum())
        codebook_used = (avg_probs > 0).sum()
        info = {"perplexity": perplexity, "codebook_used": codebook_used}
        return z_q, vq_loss, info


class ConvVQVAE3D_SimVQ(ConvVQVAE3D):
    """:class:`~models.vae3d_vq.ConvVQVAE3D` with the SimVQ quantizer swapped in.

    Encoder, decoder, skip handling and ``reconstruct``/``encode``/``decode`` are
    inherited verbatim; only ``self.quantizer`` is replaced, so the SimVQ model is
    scored by the existing 3D correlation/diagnosis scripts unchanged.
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
    ) -> None:
        # Build the full VQ-VAE (standard quantizer), then swap in SimVQ. Passing
        # vq_mode="standard" is irrelevant — the quantizer is replaced below.
        super().__init__(
            base_channels=base_channels,
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            commitment_cost=commitment_cost,
            input_size=input_size,
            in_channels=in_channels,
            use_skip=use_skip,
            vq_mode="standard",
        )
        self.vq_mode = "simvq"
        self.quantizer = SimVQuantizer3D(
            self.num_embeddings, self.embedding_dim, self.commitment_cost
        )
