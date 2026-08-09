"""Tests for the shape-VAE model and its losses.

No real RAOS data is needed: everything runs on synthetic (B, 1, 128, 128)
tensors on the CPU, so the suite is fast and GPU-independent. The tests pin the
behaviours the training loop relies on: exact output shapes, stochastic-vs-
deterministic sampling, the closed-form KL, that a matched reconstruction is
cheap, that the loss back-propagates, encode/decode round-trips, and that the
decoder actually uses its latent (a guard against a degenerate constant decoder).
"""
from __future__ import annotations

import torch

from losses.vae_losses import kl_divergence, reconstruction_loss, vae_loss
from models.vae import ConvVAE

LATENT_DIM = 32
INPUT_SIZE = (128, 128)


def make_model() -> ConvVAE:
    return ConvVAE(latent_dim=LATENT_DIM, base_channels=32, input_size=INPUT_SIZE)


def make_batch(batch: int = 4) -> torch.Tensor:
    """A batch of random binary masks, shape (B, 1, 128, 128)."""
    torch.manual_seed(0)
    return (torch.rand(batch, 1, *INPUT_SIZE) > 0.5).float()


def test_forward_shapes_are_correct() -> None:
    model = make_model()
    x = make_batch(4)
    logits, mu, logvar = model(x)
    assert logits.shape == (4, 1, *INPUT_SIZE)
    assert mu.shape == (4, LATENT_DIM)
    assert logvar.shape == (4, LATENT_DIM)


def test_reparameterize_stochastic_in_train_deterministic_in_eval() -> None:
    model = make_model()
    mu = torch.zeros(4, LATENT_DIM)
    logvar = torch.zeros(4, LATENT_DIM)  # std = 1, non-degenerate

    model.train()
    z1 = model.reparameterize(mu, logvar)
    z2 = model.reparameterize(mu, logvar)
    assert not torch.allclose(z1, z2)  # sampling adds noise

    model.eval()
    z_eval = model.reparameterize(mu, logvar)
    assert torch.allclose(z_eval, mu)  # deterministic: z = mu


def test_forward_is_deterministic_in_eval() -> None:
    model = make_model().eval()
    x = make_batch(3)
    with torch.no_grad():
        a, _, _ = model(x)
        b, _, _ = model(x)
    assert torch.allclose(a, b)


def test_kl_is_nonnegative_and_zero_at_standard_normal() -> None:
    mu = torch.randn(8, LATENT_DIM)
    logvar = torch.randn(8, LATENT_DIM)
    assert kl_divergence(mu, logvar).item() >= 0.0

    zero_kl = kl_divergence(torch.zeros(8, LATENT_DIM), torch.zeros(8, LATENT_DIM))
    assert torch.allclose(zero_kl, torch.tensor(0.0), atol=1e-6)


def test_reconstruction_loss_low_when_logits_match_target() -> None:
    target = make_batch(4)
    # Confident-correct logits: large + where target is 1, large - where 0.
    good = torch.where(target > 0.5, 10.0, -10.0)
    bad = torch.where(target > 0.5, -10.0, 10.0)
    good_recon, _, _ = reconstruction_loss(good, target)
    bad_recon, _, _ = reconstruction_loss(bad, target)
    assert good_recon.item() < 0.05
    assert bad_recon.item() > good_recon.item()


def test_vae_loss_backpropagates() -> None:
    model = make_model().train()
    x = make_batch(4)
    logits, mu, logvar = model(x)
    terms = vae_loss(logits, x, mu, logvar, beta=0.1, dice_weight=1.0)
    terms["total"].backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert any(g is not None and torch.any(g != 0) for g in grads)


def test_encode_decode_roundtrip_shapes() -> None:
    model = make_model().eval()
    x = make_batch(5)
    mu = model.encode(x)
    assert mu.shape == (5, LATENT_DIM)
    logits = model.decode(mu)
    assert logits.shape == (5, 1, *INPUT_SIZE)


def test_decoder_uses_latent_not_constant_output() -> None:
    """Guard against a degenerate decoder that ignores z (posterior-collapse
    failure mode): two different latents must give different reconstructions."""
    model = make_model().eval()
    torch.manual_seed(1)
    z_a = torch.randn(1, LATENT_DIM)
    z_b = torch.randn(1, LATENT_DIM)
    with torch.no_grad():
        out_a = model.decode(z_a)
        out_b = model.decode(z_b)
    assert not torch.allclose(out_a, out_b)


def test_input_size_indivisible_raises() -> None:
    try:
        ConvVAE(latent_dim=8, base_channels=8, input_size=(130, 130))
    except ValueError:
        return
    raise AssertionError("expected ValueError for non-divisible input_size")
