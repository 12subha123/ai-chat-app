import torch


class DiffusionProcess:
    """
    Implements the forward diffusion process.

    The forward process gradually adds Gaussian noise
    to a clean image over multiple timesteps.
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

        self.sqrt_alpha_cumprod = torch.sqrt(
            self.alpha_cumprod
        )

        self.sqrt_one_minus_alpha_cumprod = torch.sqrt(
            1.0 - self.alpha_cumprod
        )

    def add_noise(
        self,
        images: torch.Tensor,
        timesteps: torch.Tensor,
        noise: torch.Tensor | None = None,
    ):
        """
        Add noise to images according to the
        selected diffusion timestep.
        """

        if noise is None:
            noise = torch.randn_like(images)

        sqrt_alpha = self.sqrt_alpha_cumprod[
            timesteps
        ]

        sqrt_one_minus_alpha = (
            self.sqrt_one_minus_alpha_cumprod[
                timesteps
            ]
        )

        sqrt_alpha = sqrt_alpha.view(
            -1, 1, 1, 1
        )

        sqrt_one_minus_alpha = (
            sqrt_one_minus_alpha.view(
                -1, 1, 1, 1
            )
        )

        noisy_images = (
            sqrt_alpha * images
            + sqrt_one_minus_alpha * noise
        )

        return noisy_images, noise