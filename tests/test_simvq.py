"""Unit tests for the 3D SimVQ-VAE (Phase 12), all on tiny synthetic tensors.

Guard the SimVQ contract on top of the shared VQ contract: the base codebook C is
FROZEN (no gradient) while the single linear layer W is the only trained codebook
parameter; the quantizer still emits exact effective-codebook rows through a
differentiable straight-through path; and the ConvVQVAE3D_SimVQ subclass inherits
the encoder/decoder unchanged so forward/reconstruct/roundtrip all behave.
"""
from __future__ import annotations

import torch

from losses.vae_losses import vq_loss
from models.vae3d_simvq import ConvVQVAE3D_SimVQ, SimVQuantizer3D

INPUT = (16, 16, 16)


def _model(use_skip: bool = False) -> ConvVQVAE3D_SimVQ:
    return ConvVQVAE3D_SimVQ(
        base_channels=8,
        num_embeddings=64,
        embedding_dim=16,
        commitment_cost=0.25,
        input_size=INPUT,
        use_skip=use_skip,
    )


def _batch(b: int = 2) -> torch.Tensor:
    torch.manual_seed(0)
    return (torch.rand(b, 1, *INPUT) > 0.5).float()


# -- quantizer: SimVQ structure ---------------------------------------------
def test_base_codebook_is_frozen() -> None:
    """C (self.codebook) must be non-trainable; W (self.proj) must be trainable."""
    vq = SimVQuantizer3D(num_embeddings=8, embedding_dim=4, commitment_cost=0.25)
    assert vq.codebook.weight.requires_grad is False
    assert vq.proj.weight.requires_grad is True


def test_only_W_receives_gradient() -> None:
    """A backward pass must leave C's grad None but give W a real gradient."""
    vq = SimVQuantizer3D(num_embeddings=8, embedding_dim=4, commitment_cost=0.25)
    z = torch.randn(2, 4, 3, 3, 3, requires_grad=True)
    z_q, loss, _ = vq(z)
    (loss + z_q.sum()).backward()
    assert vq.codebook.weight.grad is None            # frozen base gets no grad
    assert vq.proj.weight.grad is not None            # the linear map is trained
    assert vq.proj.weight.grad.abs().sum().item() > 0.0


def test_quantizer_output_comes_from_effective_codebook() -> None:
    """z_q values must equal exact rows of the EFFECTIVE codebook C@W."""
    vq = SimVQuantizer3D(num_embeddings=8, embedding_dim=4, commitment_cost=0.25)
    z = torch.randn(2, 4, 3, 3, 3)
    z_q, loss, _ = vq(z)
    assert z_q.shape == z.shape
    assert torch.isfinite(loss) and loss.item() >= 0.0
    flat = z.permute(0, 2, 3, 4, 1).reshape(-1, 4)
    w = vq.effective_codebook()
    dist = flat.pow(2).sum(1, keepdim=True) - 2 * flat @ w.t() + w.pow(2).sum(1)
    idx = dist.argmin(1)
    expected = w[idx].view(2, 3, 3, 3, 4).permute(0, 4, 1, 2, 3)
    assert torch.allclose(z_q, expected, atol=1e-6)


def test_effective_codebook_depends_on_W() -> None:
    """Changing W must change the effective codebook (it is not the frozen C)."""
    vq = SimVQuantizer3D(num_embeddings=8, embedding_dim=4, commitment_cost=0.25)
    before = vq.effective_codebook().detach().clone()
    with torch.no_grad():
        vq.proj.weight.add_(1.0)
    after = vq.effective_codebook().detach()
    assert not torch.allclose(before, after)


def test_quantizer_perplexity_and_usage() -> None:
    vq = SimVQuantizer3D(num_embeddings=16, embedding_dim=4, commitment_cost=0.25)
    z = torch.randn(4, 4, 2, 2, 2)
    _, _, info = vq(z)
    assert info["perplexity"].item() > 0.0
    assert 1 <= int(info["codebook_used"].item()) <= 16


# -- model forward / reconstruct / roundtrip --------------------------------
def test_forward_shapes_and_vq_loss() -> None:
    model = _model()
    x = _batch()
    logits, vq_term, info = model(x)
    assert logits.shape == x.shape
    assert torch.isfinite(vq_term) and vq_term.item() >= 0.0
    assert "perplexity" in info and "codebook_used" in info


def test_reconstruct_is_probability() -> None:
    model = _model()
    x = _batch()
    probs = model.reconstruct(x)
    assert probs.shape == x.shape
    assert float(probs.min()) >= 0.0 and float(probs.max()) <= 1.0


def test_encode_decode_roundtrip_shapes() -> None:
    model = _model(use_skip=False)
    x = _batch()
    code = model.encode(x)
    assert code.shape == (x.size(0), 16)              # embedding_dim * 1*1*1
    z_q, _, _ = model.quantize(x)
    logits = model.decode(z_q)
    assert logits.shape == x.shape


def test_backprop_reaches_encoder() -> None:
    """Straight-through estimator must let gradients reach encoder weights."""
    model = _model()
    x = _batch()
    logits, vq_term, _ = model(x)
    terms = vq_loss(logits, x, vq_term)
    terms["total"].backward()
    first_conv = model.enc_stages[0][0]
    assert first_conv.weight.grad is not None
    assert torch.isfinite(first_conv.weight.grad).all()
    assert first_conv.weight.grad.abs().sum().item() > 0.0


def test_base_codebook_frozen_after_optimizer_step() -> None:
    """A real Adam step over trainable params must not move the frozen base C."""
    model = _model()
    x = _batch()
    c_before = model.quantizer.codebook.weight.detach().clone()
    opt = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=1e-2)
    logits, vq_term, _ = model(x)
    vq_loss(logits, x, vq_term)["total"].backward()
    opt.step()
    assert torch.allclose(model.quantizer.codebook.weight, c_before)


def test_vq_mode_label_is_simvq() -> None:
    model = _model()
    assert model.vq_mode == "simvq"
    assert model.quantizer.vq_mode == "simvq"
