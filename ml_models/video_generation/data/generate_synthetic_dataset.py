from pathlib import Path
import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIRECTORY = PROJECT_ROOT / "video_dataset_source"

VIDEO_COUNT = 20
FRAME_COUNT = 16
FRAME_SIZE = 64
FPS = 8

def generate_video(video_index):
    output_path = OUTPUT_DIRECTORY / f"synthetic_{video_index:03d}.mp4"

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    writer = cv2.VideoWriter(
        str(output_path),
        fourcc,
        FPS,
        (FRAME_SIZE, FRAME_SIZE),
    )

    if not writer.isOpened():
        raise RuntimeError(
            f"Could not create video: {output_path}"
        )

    start_x = np.random.randint(8, FRAME_SIZE - 24)
    start_y = np.random.randint(8, FRAME_SIZE - 24)

    velocity_x = np.random.choice([-2, -1, 1, 2])
    velocity_y = np.random.choice([-2, -1, 1, 2])

    color = tuple(
        int(value)
        for value in np.random.randint(80, 256, size=3)
    )

    x = start_x
    y = start_y

    for _ in range(FRAME_COUNT):
        frame = np.zeros(
            (FRAME_SIZE, FRAME_SIZE, 3),
            dtype=np.uint8,
        )

        cv2.rectangle(
            frame,
            (x, y),
            (x + 16, y + 16),
            color,
            -1,
        )

        writer.write(frame)

        x += velocity_x
        y += velocity_y

        if x <= 0 or x >= FRAME_SIZE - 17:
            velocity_x *= -1

        if y <= 0 or y >= FRAME_SIZE - 17:
            velocity_y *= -1

    writer.release()

def main():
    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    for video_index in range(VIDEO_COUNT):
        generate_video(video_index)

    print(
        f"Generated {VIDEO_COUNT} synthetic videos."
    )


if __name__ == "__main__":
    main()
