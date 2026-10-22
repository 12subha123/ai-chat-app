import torch
import torch.nn.functional as F


def diffusion_training_step(
    model,
    diffusion,
    images,
    optimizer,
):
    """
    Perform one training step for the diffusion model.
    """

    model.train()

    batch_size = images.shape[0]

    timesteps = torch.randint(
        0,
        diffusion.timesteps,
        (batch_size,),
        device=images.device,
    )

    noisy_images, noise = diffusion.add_noise(
        images,
        timesteps,
    )

    predicted_noise = model(
        noisy_images,
        timesteps,
    )

    loss = F.mse_loss(
        predicted_noise,
        noise,
    )

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    return loss.item()