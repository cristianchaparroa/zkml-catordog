import { DebugContracts } from "./_components/DebugContracts";
import type { NextPage } from "next";
import { getMetadata } from "~~/utils/scaffold-stark/getMetadata";

export const metadata = getMetadata({
  title: "Debug Contracts",
  description: "Debug your deployed 🏗 zkml-catordog contracts in an easy way",
});

const Debug: NextPage = () => {
  return <DebugContracts />;
};

export default Debug;
