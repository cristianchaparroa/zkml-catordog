import { useState } from "react";
import Image from "next/image";

export function AnalyzeImage(params: any) {
  // console.log(JSON.stringify(params));

  const [selectedImage, setSelectedImage] = useState<string | null>(null);
  const [isLoading, setLoading] = useState(false);
  const [diagnose, setDiagnose] = useState("");
  const [ipfsHash, setIpfsHash] = useState("");

  const handleAnalyze = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();

    reader.onload = (e) => {
      // Set image as a data URL
      const imageDataUrl = e.target?.result as string;
      setSelectedImage(imageDataUrl);
    };

    // Read the file as a data URL
    reader.readAsDataURL(file);

    setLoading(true);
  };

  const handleNewImage = (event: any) => {
    const fileInput = document.getElementById("fileInput") as HTMLInputElement;
    fileInput.click();
  };

  return (
    <div className="mt-10 flex flex-col bg-base-100 relative text-[12px] px-10 py-10 text-center items-center max-w-xs rounded-3xl border border-gradient">
      <div className="trapeze"></div>
      {!selectedImage && (
        <>
          <svg
            xmlns="http://www.w3.org/2000/svg"
            viewBox="0 0 20 20"
            className="size-9 fill-current"
          >
            <path
              fillRule="evenodd"
              d="M9 3.5a5.5 5.5 0 1 0 0 11 5.5 5.5 0 0 0 0-11ZM2 9a7 7 0 1 1 12.452 4.391l3.328 3.329a.75.75 0 1 1-1.06 1.06l-3.329-3.328A7 7 0 0 1 2 9Z"
              clipRule="evenodd"
            />
          </svg>
          <p className="mt-2 mb-8 text-base">
            Please, choose a cat or dog image.
          </p>
        </>
      )}
      <div className="wrapper">
        {selectedImage && (
          <>
            <p className="mb-2">Selected image: </p>
            <Image
              src={selectedImage}
              width={200}
              height={200}
              alt="Selected image"
              className="mb-7 rounded-md"
            />
          </>
        )}
        {isLoading ? (
          <p>Processing uploaded image...</p>
        ) : (
          <>
            <input
              type="file"
              accept="image/*"
              id="fileInput"
              style={{ display: "none" }} // Hide the input element
              onChange={handleAnalyze}
            />
            <button className="btn btn-accent" onClick={handleNewImage}>
              Analyze Image
            </button>
          </>
        )}
      </div>
    </div>
  );
}
