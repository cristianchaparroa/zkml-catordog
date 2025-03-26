import { useState } from "react";
import Image from "next/image";

export function AnalyzeImage(params: any) {
  const [selectedImage, setSelectedImage] = useState<string | null>(null);
  const [isLoading, setLoading] = useState(false);
  const [uploadedImagePath, setUploadedImagePath] = useState<string | null>(
    null
  );
  const [analysisData, setAnalysisData] = useState<any>(null);
  const [verificationData, setVerificationData] = useState<any>(null);
  const [isVerifying, setIsVerifying] = useState(false);
  const [isVerified, setIsVerified] = useState(false);
  const [isValid, setIsValid] = useState(false);

  const handleAnalyze = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    reset();
    setLoading(true);
    showImage(file);

    try {
      // Call the analyze endpoint with the ORIGINAL file
      const analysisFormData = new FormData();
      analysisFormData.append("file", file); // Use the original file object directly

      const analysisResponse = await fetch(
        `${params.zkml_backend_url}/images`,
        {
          method: "POST",
          body: analysisFormData,
        }
      );

      if (!analysisResponse.ok) {
        throw new Error("There was an error trying to analyze the image");
      }

      const returnedAnalysisData = await analysisResponse.json();
      setAnalysisData(returnedAnalysisData);

      //Upload image and analysis
      const formData = new FormData();
      formData.append("image", file);
      formData.append("analysisData", JSON.stringify(returnedAnalysisData));

      const response = await fetch("/api/upload-image", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error("There was an error trying to upload the image");
      }

      const data = await response.json();
      setUploadedImagePath(data.fullPath);

      setLoading(false);
    } catch (error) {
      console.error("Error:", error);
      setLoading(false);
    }
  };

  const showImage = (file: any) => {
    const reader = new FileReader();

    reader.onload = (e) => {
      // Set image as a data URL
      const imageDataUrl = e.target?.result as string;
      setSelectedImage(imageDataUrl);
    };

    reader.readAsDataURL(file);
  };

  const handleNewImage = (event: any) => {
    const fileInput = document.getElementById("fileInput") as HTMLInputElement;
    fileInput.click();
  };

  const handleCancel = (event: any) => {
    reset();
  };

  const handleVerify = async (event: any) => {
    if (!analysisData) return;

    try {
      setIsValid(false);
      setIsVerifying(true);

      const formData = new FormData();
      formData.append("id", analysisData.id);

      const response = await fetch(`${params.zkml_verifier_url}/verifies`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error("There was an error trying to verify the proof");
      }

      const data = await response.json();

      if (data.verified) {
        setIsValid(true);
      }

      setIsVerified(true);
      setIsVerifying(false);
    } catch (error) {
      console.error("Error:", error);
      setLoading(false);
    }
  };

  const reset = () => {
    setLoading(false);
    setSelectedImage(null);
    setUploadedImagePath(null);
    setAnalysisData(null);
    setVerificationData(null);
    setIsVerifying(false);
    setIsVerified(false);
    setIsValid(false);
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
            <p className="mb-2 text-lg">Selected image: </p>
            <Image
              src={selectedImage}
              width={200}
              height={200}
              alt="Selected image"
              className="mb-2 rounded-md"
            />
          </>
        )}
        {analysisData && (
          <>
            <p className="mt-0 mb-0 text-base">
              It is a: {analysisData.class_name}
            </p>
            <p className="mt-0 mb-2 text-base">
              Confidence: {parseFloat(analysisData.confidence).toFixed(4)}
            </p>
            {!isVerified && !isVerifying ? (
              <button className="mb-7 btn btn-neutral" onClick={handleVerify}>
                Verify
              </button>
            ) : (
              <>
                {isVerifying && (
                  <p className="mb-2 text-base">Verifying proof...</p>
                )}
                {isVerified && (
                  <>
                    (isValid) ? (
                    <p className="mb-2 text-base">Proof is valid!</p>) : (
                    <p className="mb-2 text-base">Proof is NOT valid.</p>)
                  </>
                )}
              </>
            )}
          </>
        )}
        {isLoading ? (
          <>
            <p className="mb-2 text-base">Processing uploaded image...</p>
            <button className="btn btn-cancel" onClick={handleCancel}>
              Cancel
            </button>
          </>
        ) : (
          <div className="mt-3">
            <input
              type="file"
              accept="image/*"
              id="fileInput"
              style={{ display: "none" }} // Hide the input element
              onChange={handleAnalyze}
            />
            <button className="btn btn-accent" onClick={handleNewImage}>
              {selectedImage ? "Analyze a new image" : "Analyze image"}
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
