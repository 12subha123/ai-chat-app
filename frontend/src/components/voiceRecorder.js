import React, { useRef, useState } from "react";

function VoiceRecorder({ onTranscription }) {
  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);

  const [isRecording, setIsRecording] = useState(false);
  const [error, setError] = useState("");
  const [transcribedText, setTranscribedText] = useState("");

  const startRecording = async () => {
    try {
      setError("");
      setTranscribedText("");

      const stream = await navigator.mediaDevices.getUserMedia({
        audio: true,
      });

      const mediaRecorder = new MediaRecorder(stream);

      mediaRecorderRef.current = mediaRecorder;
      audioChunksRef.current = [];

      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };

      mediaRecorder.onstop = async () => {
        const audioBlob = new Blob(
          audioChunksRef.current,
          {
            type: mediaRecorder.mimeType,
          }
        );

        stream.getTracks().forEach((track) => {
          track.stop();
        });

        try {
          const formData = new FormData();

          formData.append(
            "file",
            audioBlob,
            "voice.webm"
          );

          const response = await fetch(
            "http://127.0.0.1:8001/voice/transcribe",
            {
              method: "POST",
              body: formData,
            }
          );

          if (!response.ok) {
            throw new Error(
              `Server error: ${response.status}`
            );
          }

          const data = await response.json();

          setTranscribedText(data.text || "");

          if (onTranscription) {
            onTranscription(data.text || "");
          }
        } catch (err) {
          console.error("Transcription error:", err);

          setError(
            "Could not transcribe the recording."
          );
        }
      };

      mediaRecorder.start();
      setIsRecording(true);
    } catch (err) {
      console.error("Microphone error:", err);

      setError(
        "Microphone access was denied or is unavailable."
      );
    }
  };

  const stopRecording = () => {
    const mediaRecorder = mediaRecorderRef.current;

    if (!mediaRecorder) {
      return;
    }

    if (mediaRecorder.state !== "inactive") {
      mediaRecorder.stop();
    }

    setIsRecording(false);
  };

  return (
    <div>
      {!isRecording ? (
        <button
          type="button"
          onClick={startRecording}
        >
          🎤 Start Voice
        </button>
      ) : (
        <button
          type="button"
          onClick={stopRecording}
        >
          ⏹ Stop Recording
        </button>
      )}

      {transcribedText && (
        <p>
          <strong>You said:</strong>{" "}
          {transcribedText}
        </p>
      )}

      {error && (
        <p>
          {error}
        </p>
      )}
    </div>
  );
}

export default VoiceRecorder;