from pathlib import Path

import cv2


PROJECT_ROOT = Path(__file__).resolve().parents[3]

FRAME_DIRECTORY = (
    PROJECT_ROOT
    / "ml_models"
    / "video_generation"
    / "inference"
    / "generated_frames"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "ml_models"
    / "video_generation"
    / "inference"
    / "generated_video.mp4"
)

FRAME_RATE = 8


def main():
    frame_paths = sorted(
        FRAME_DIRECTORY.glob(
            "generated_frame_*.png"
        )
    )

    if not frame_paths:
        raise FileNotFoundError(
            f"No generated frames found in: {FRAME_DIRECTORY}"
        )

    first_frame = cv2.imread(
        str(frame_paths[0])
    )

    if first_frame is None:
        raise RuntimeError(
            f"Could not read frame: {frame_paths[0]}"
        )

    height, width = first_frame.shape[:2]

    fourcc = cv2.VideoWriter_fourcc(
        *"mp4v"
    )

    writer = cv2.VideoWriter(
        str(OUTPUT_PATH),
        fourcc,
        FRAME_RATE,
        (width, height),
    )

    if not writer.isOpened():
        raise RuntimeError(
            f"Could not create video: {OUTPUT_PATH}"
        )

    for frame_path in frame_paths:
        frame = cv2.imread(
            str(frame_path)
        )

        if frame is None:
            raise RuntimeError(
                f"Could not read frame: {frame_path}"
            )

        writer.write(frame)

    writer.release()

    print(
        f"Video created successfully: {OUTPUT_PATH}"
    )

    print(
        f"Frames used: {len(frame_paths)}"
    )

    print(
        f"Frame rate: {FRAME_RATE} FPS"
    )


if __name__ == "__main__":
    main()
