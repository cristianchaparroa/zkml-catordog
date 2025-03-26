"use client";
import Link from "next/link";
import { ConnectedAddress } from "~~/components/ConnectedAddress";
import { AnalyzeImage } from "~~/components/AnalyzeImage";

const ZKML_BACKEND_URL = process.env.NEXT_PUBLIC_ZKML_BACKEND_URL;
const ZKML_VERIFIER_URL = process.env.NEXT_PUBLIC_ZKML_VERIFIER_URL;

const Home = () => {
  return (
    <div className="flex items-center flex-col flex-grow pt-10">
      <div className="px-5">
        <h1 className="text-center">
          <span className="block text-2xl mb-2">Welcome to</span>
          <span className="block text-4xl font-bold">zkml CatOrDog</span>
        </h1>
        {/* <ConnectedAddress /> */}
        <AnalyzeImage
          zkml_backend_url={ZKML_BACKEND_URL}
          zkml_verifier_url={ZKML_VERIFIER_URL}
        />
      </div>
    </div>
  );
};

export default Home;
