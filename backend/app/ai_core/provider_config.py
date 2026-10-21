from dataclasses import dataclass


@dataclass
class ProviderConfig:
    llm: str = "gemini"
    web_answer: str = "gemini"
    knowledge: str = "internal_rag"
    web: str = "tavily"
    voice: str = "local_voice"
    image: str = "local_diffusion"
    video: str = "local_video"

    def validate(self) -> None:
        providers = {
            "llm": self.llm,
            "web_answer": self.web_answer,
            "knowledge": self.knowledge,
            "web": self.web,
            "voice": self.voice,
            "image": self.image,
            "video": self.video,
        }

        for capability, provider in providers.items():
            if (
                not isinstance(provider, str)
                or not provider.strip()
            ):
                raise ValueError(
                    f"Provider configuration for "
                    f"'{capability}' cannot be empty."
                )
