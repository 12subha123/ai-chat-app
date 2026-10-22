const API_BASE_URL = "http://127.0.0.1:8001";

export async function uploadFile(file) {
  const formData = new FormData();

  formData.append("file", file);

  const response = await fetch(
    `${API_BASE_URL}/uploads`,
    {
      method: "POST",
      body: formData,
    }
  );

  if (!response.ok) {
    throw new Error(
      `Upload failed. Backend returned status ${response.status}`
    );
  }

  return response.json();
}
