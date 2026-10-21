from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]

SOURCE_DIRECTORY = PROJECT_ROOT / "video_dataset_source"

OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "ml_models"
    / "video_generation"
    / "data"
    / "train"
)

FRAME_SIZE = 64
FRAMES_PER_CLIP = 8


def extract_clips():
    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    video_files = list(
        SOURCE_DIRECTORY.glob("*.mp4")
    )

    if not video_files:
        raise FileNotFoundError(
            f"No .mp4 videos found in: {SOURCE_DIRECTORY}"
        )

    clip_index = 0

    for video_path in video_files:
        capture = cv2.VideoCapture(
            str(video_path)
        )

        frames = []

        while True:
            success, frame = capture.read()

            if not success:
                break

            frame = cv2.resize(
                frame,
                (FRAME_SIZE, FRAME_SIZE),
            )

            frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB,
            )

            frames.append(frame)

            if len(frames) == FRAMES_PER_CLIP:
                output_path = (
                    OUTPUT_DIRECTORY
                    / f"clip_{clip_index:05d}.npy"
                )

                np.save(
                    output_path,
                    np.array(frames),
                )

                clip_index += 1
                frames = []

        capture.release()

    print(
        f"Created {clip_index} training clips."
    )


if __name__ == "__main__":
    extract_clips()
