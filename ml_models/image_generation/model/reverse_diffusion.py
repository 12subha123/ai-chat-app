import torch


class ReverseDiffusionProcess:
    """
    Implements the DDPM reverse diffusion process.

    Starting from random Gaussian noise, the process
    gradually reconstructs an image by removing predicted noise.
    """

    def __init__(
        self,
        timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
    ):
        self.timesteps = timesteps

        self.betas = torch.linspace(
            beta_start,
            beta_end,
            timesteps,
            dtype=torch.float32,
        )

        self.alphas = 1.0 - self.betas

        self.alpha_cumprod = torch.cumprod(
            self.alphas,
            dim=0,
        )

        self.alpha_cumprod_prev = torch.cat(
            [
                torch.tensor([1.0]),
                self.alpha_cumprod[:-1],
            ]
        )

        self.posterior_variance = (
            self.betas
            * (
                1.0
                - self.alpha_cumprod_prev
            )
            / (
                1.0
                - self.alpha_cumprod
            )
        )

    @torch.no_grad()
    def step(
        self,
        model,
        noisy_images: torch.Tensor,
        timesteps: torch.Tensor,
    ):
        """
        Perform one DDPM reverse-diffusion step.
        """

        predicted_noise = model(
            noisy_images,
            timesteps,
        )

        beta = self.betas[timesteps]
        alpha = self.alphas[timesteps]
        alpha_cumprod = self.alpha_cumprod[timesteps]
        alpha_cumprod_prev = (
            self.alpha_cumprod_prev[timesteps]
        )

        beta = beta.view(-1, 1, 1, 1)
        alpha = alpha.view(-1, 1, 1, 1)
        alpha_cumprod = alpha_cumprod.view(
            -1, 1, 1, 1
        )
        alpha_cumprod_prev = (
            alpha_cumprod_prev.view(
                -1, 1, 1, 1
            )
        )

        predicted_original = (
            noisy_images
            - torch.sqrt(
                1.0 - alpha_cumprod
            ) * predicted_noise
        ) / torch.sqrt(alpha_cumprod)

        predicted_original = torch.clamp(
            predicted_original,
            -1.0,
            1.0,
        )

        posterior_mean = (
            (
                torch.sqrt(
                    alpha_cumprod_prev
                )
                * beta
                / (
                    1.0
                    - alpha_cumprod
                )
            )
            * predicted_original
            +
            (
                torch.sqrt(alpha)
                * (
                    1.0
                    - alpha_cumprod_prev
                )
                / (
                    1.0
                    - alpha_cumprod
                )
            )
            * noisy_images
        )

        posterior_variance = (
            self.posterior_variance[timesteps]
            .view(-1, 1, 1, 1)
        )

        noise = torch.randn_like(
            noisy_images
        )

        nonzero_mask = (
            timesteps != 0
        ).float().view(
            -1, 1, 1, 1
        )

        return (
            posterior_mean
            + nonzero_mask
            * torch.sqrt(
                posterior_variance
            )
            * noise
        )