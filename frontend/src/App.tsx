// file to create a simple file uploader and predict the image
import { ChangeEvent, useState } from "react";
import axios from "axios";
import "./App.css";

function App() {
  const [image, setImage] = useState<File | null>(null);
  const [text, setText] = useState<string>("");
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);

  // function for setting the image selected
  const changeImage = (e: ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files?.[0];
    if (files) {
      setImage(files);
      setPreviewUrl(URL.createObjectURL(files));
    } else {
      setImage(null);
      setPreviewUrl(null);
    }
  };

  // for passing the image to the backend
  const predict = async () => {
    if (!image) {
      alert("Please select an image first!");
      return;
    }

    const formData = new FormData();
    formData.append("image", image);

    try {
      const response = await axios.post(
        `http://localhost:4000/predict`,
        formData,
        {
          headers: {
            "Content-Type": "multipart/form-data",
          },
        }
      );
      if (response.status === 200) {
        setText(response.data.data);
      }
    } catch (error) {
      console.error("error", error);
      setText("Prediction failed. Please check the backend.");
    }
  };

  return (
    <div className="container">
      <div className="upload-section">
        <h1>Upload your image here</h1>
        <input type="file" onChange={changeImage} accept="image/*" />
      </div>
      <div className="preview-section">
        <p>Preview:</p>
        {previewUrl && (
          <img src={previewUrl} alt="preview" className="preview-image" />
        )}
        {!previewUrl && <p>No image selected.</p>}
      </div>
      <button onClick={predict} disabled={!image}>
        Predict the image
      </button>
      <div className="output-section">
        <p>Output: {text}</p>
      </div>
    </div>
  );
}

export default App;
