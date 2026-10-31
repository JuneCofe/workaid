import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";

// Demo builds (GitHub Pages) are served from /<repo>/; the full app is served from /.
export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), "");
  const demo = env.VITE_DEMO === "1";
  return {
    root: "frontend",
    base: demo ? env.VITE_BASE || "/" : "/",
    plugins: [react()],
    publicDir: "public",
    build: {
      outDir: "../dist",
      emptyOutDir: true,
      chunkSizeWarningLimit: 600,
    },
    server: {
      proxy: { "/api": "http://127.0.0.1:8081" },
    },
  };
});
