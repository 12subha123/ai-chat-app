import { useRef, useState } from "react";
import { uploadFile } from "../services/attachment";

function DocumentUpload() {
  const fileInputRef = useRef(null);
  const [status, setStatus] = useState("");

  async function handleFileChange(event) {
    const file = event.target.files?.[0];

    if (!file) {
      return;
    }

    try {
      setStatus("Uploading...");

      const result = await uploadFile(file);

      console.log("Upload result:", result);

      setStatus(`Uploaded: ${result.filename}`);
    } catch (error) {
      console.error("Upload error:", error);

      setStatus("Upload failed.");
    }

    event.target.value = "";
  }

  return (
    <div>
      <button
        type="button"
        onClick={() => fileInputRef.current?.click()}
      >
        ?? Upload Document
      </button>

      <input
        ref={fileInputRef}
        type="file"
        accept=".pdf,.docx,.txt,.md,.jpg,.jpeg,.png,.webp"
        onChange={handleFileChange}
        style={{ display: "none" }}
      />

      {status && (
        <div>
          {status}
        </div>
      )}
    </div>
  );
}

export default DocumentUpload;
