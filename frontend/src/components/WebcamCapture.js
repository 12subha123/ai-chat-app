import { useRef, useState } from "react";
import { uploadFile } from "../services/attachment";

function WebcamCapture({ onUpload }) {
  const videoRef = useRef(null);
  const streamRef = useRef(null);

  const [cameraOpen, setCameraOpen] = useState(false);
  const [status, setStatus] = useState("");

  async function startCamera() {
    try {
      const stream =
        await navigator.mediaDevices.getUserMedia({
          video: true,
          audio: false,
        });

      streamRef.current = stream;

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
      }

      setCameraOpen(true);
      setStatus("");
    } catch (error) {
      console.error("Camera error:", error);
      setStatus(
        "Could not access the camera. Please allow camera permission."
      );
    }
  }

  function stopCamera() {
    if (streamRef.current) {
      streamRef.current
        .getTracks()
        .forEach((track) => track.stop());

      streamRef.current = null;
    }

    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }

    setCameraOpen(false);
  }

  async function captureImage() {
    if (!videoRef.current) {
      return;
    }

    const video = videoRef.current;

    const canvas = document.createElement("canvas");

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    const context = canvas.getContext("2d");

    context.drawImage(
      video,
      0,
      0,
      canvas.width,
      canvas.height
    );

    canvas.toBlob(
      async (blob) => {
        if (!blob) {
          setStatus("Could not capture image.");
          return;
        }

        try {
          setStatus("Uploading captured image...");

          const file = new File(
            [blob],
            `webcam-document-${Date.now()}.jpg`,
            {
              type: "image/jpeg",
            }
          );

          const result = await uploadFile(file);

          console.log(
            "Webcam upload result:",
            result
          );

          setStatus(
            `Captured and uploaded: ${result.filename}`
          );

          if (onUpload) {
            onUpload(result);
          }
        } catch (error) {
          console.error(
            "Webcam upload error:",
            error
          );

          setStatus("Failed to upload captured image.");
        }
      },
      "image/jpeg",
      0.95
    );
  }

  return (
    <div>
      {!cameraOpen && (
        <button
          type="button"
          onClick={startCamera}
        >
          ?? Open Camera
        </button>
      )}

      {cameraOpen && (
        <div>
          <video
            ref={videoRef}
            autoPlay
            playsInline
            style={{
              width: "100%",
              maxWidth: "640px",
            }}
          />

          <div>
            <button
              type="button"
              onClick={captureImage}
            >
              ?? Capture Document
            </button>

            <button
              type="button"
              onClick={stopCamera}
            >
              ? Close Camera
            </button>
          </div>
        </div>
      )}

      {status && (
        <div>
          {status}
        </div>
      )}
    </div>
  );
}

export default WebcamCapture;
