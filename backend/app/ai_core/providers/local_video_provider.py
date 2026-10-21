from pathlib import Path
import subprocess
import sys
import uuid

from app.ai_core.models import AIRequest, AIResponse


class LocalVideoProvider:
    name = "local_video"

    def __init__(self):
        self.project_root = Path(__file__).resolve().parents[4]

        self.script_path = (
            self.project_root
            / "ml_models"
            / "video_generation"
            / "inference"
            / "generate_from_noise.py"
        )

        self.frames_to_video_script = (
            self.project_root
            / "ml_models"
            / "video_generation"
            / "inference"
            / "frames_to_video.py"
        )

        self.output_directory = (
            self.project_root
            / "ml_models"
            / "video_generation"
            / "inference"
        )

    @property
    def available(self) -> bool:
        return (
            self.script_path.exists()
            and self.frames_to_video_script.exists()
        )

    def generate(self) -> str:
        if not self.available:
            raise FileNotFoundError(
                "Local video generation scripts were not found."
            )

        python_executable = sys.executable

        subprocess.run(
            [
                python_executable,
                str(self.script_path),
            ],
            cwd=str(self.project_root),
            check=True,
        )

        subprocess.run(
            [
                python_executable,
                str(self.frames_to_video_script),
            ],
            cwd=str(self.project_root),
            check=True,
        )

        generated_video = (
            self.output_directory
            / "generated_video.mp4"
        )

        if not generated_video.exists():
            raise FileNotFoundError(
                f"Generated video not found: {generated_video}"
            )

        unique_name = (
            f"generated_{uuid.uuid4().hex}.mp4"
        )

        final_path = (
            self.output_directory
            / unique_name
        )

        generated_video.replace(final_path)

        return str(final_path)

    def execute(
        self,
        request: AIRequest,
    ) -> AIResponse:
        video_path = self.generate()

        return AIResponse(
            message="Video generated successfully.",
            source="video",
            metadata={
                "video_path": video_path,
                "prompt": request.message,
            },
        )
