import type { AppProps } from "next/app";
import Navbar from "@/components/navbar";
import "@/styles/globals.css";

function MyApp({ Component, pageProps }: AppProps) {
  return (
    <>
      <Navbar />
    </>
  );
}

export default MyApp;
