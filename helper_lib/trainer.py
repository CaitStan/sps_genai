import math

import torch


def train_gan(model, data_loader, criterion, optimizer, device="cpu", epochs=10,
              *, on_epoch_end=None):
    """Train a GAN and return it, matching the Module 6 helper interface.

    criterion is BCELoss. optimizer is a dict with 'generator' and
    'discriminator' optimizers. MNIST tensors must be normalized to [-1, 1].
    Epoch statistics are also attached as model.history.
    """
    if epochs < 1:
        raise ValueError("epochs must be positive")
    if set(optimizer) != {"generator", "discriminator"}:
        raise ValueError("Provide separate generator and discriminator optimizers")
    model.to(device)
    g, d = model.generator, model.discriminator
    opt_g, opt_d = optimizer["generator"], optimizer["discriminator"]
    model.history = []
    for epoch in range(1, epochs + 1):
        model.train()
        count, g_total, d_total = 0, 0.0, 0.0
        real_total, fake_total = 0.0, 0.0
        for batch in data_loader:
            real = (batch[0] if isinstance(batch, (tuple, list)) else batch).to(device)
            batch_size = real.size(0)
            real_labels = torch.ones(batch_size, 1, device=device)
            fake_labels = torch.zeros(batch_size, 1, device=device)

            # D learns to distinguish real MNIST from detached generator output.
            opt_d.zero_grad(set_to_none=True)
            fake = g(torch.randn(batch_size, g.latent_dim, device=device))
            real_scores = d(real)
            fake_scores = d(fake.detach())
            d_loss = criterion(real_scores, real_labels) + criterion(fake_scores, fake_labels)
            d_loss.backward()
            opt_d.step()

            # G learns to make D predict real. Freeze D weights but retain the
            # differentiable path from D's output back into G's generated images.
            opt_g.zero_grad(set_to_none=True)
            d.requires_grad_(False)
            try:
                g_loss = criterion(d(fake), real_labels)
                g_loss.backward()
                opt_g.step()
            finally:
                d.requires_grad_(True)
            g_value, d_value = g_loss.item(), d_loss.item()
            if not math.isfinite(g_value + d_value):
                raise RuntimeError("Non-finite GAN loss; stop and inspect the run")
            count += batch_size
            g_total += g_value * batch_size
            d_total += d_value * batch_size
            real_total += real_scores.detach().mean().item() * batch_size
            fake_total += fake_scores.detach().mean().item() * batch_size
        if not count:
            raise ValueError("The training data loader is empty")
        row = {"epoch": epoch, "examples": count, "generator_loss": g_total / count,
               "discriminator_loss": d_total / count,
               "d_real_mean": real_total / count, "d_fake_mean": fake_total / count}
        model.history.append(row)
        print(f"Epoch {epoch}/{epochs}: G={row['generator_loss']:.4f}, "
              f"D={row['discriminator_loss']:.4f}", flush=True)
        if on_epoch_end is not None:
            on_epoch_end(model, row)
    return model
