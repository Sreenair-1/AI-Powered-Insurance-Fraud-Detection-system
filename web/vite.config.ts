import path from "node:path";
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The dev server proxies /api to the FastAPI app (uvicorn on :8000), so the
// browser talks to one origin and no CORS is needed during development.
const apiTarget = process.env.INSURESIGHT_API_URL ?? "http://127.0.0.1:8000";

export default defineConfig({
  plugins: [react()],
  resolve: { alias: { "@": path.resolve(__dirname, "src") } },
  server: {
    port: 5173,
    proxy: {
      "/api": { target: apiTarget, changeOrigin: true, rewrite: (p) => p.replace(/^\/api/, "") },
    },
  },
  preview: {
    port: 4173,
    proxy: {
      "/api": { target: apiTarget, changeOrigin: true, rewrite: (p) => p.replace(/^\/api/, "") },
    },
  },
  build: {
    chunkSizeWarningLimit: 900,
    rollupOptions: {
      output: {
        manualChunks: {
          react: ["react", "react-dom", "react-router-dom"],
          charts: ["recharts"],
          graph: ["react-force-graph-2d"],
        },
      },
    },
  },
});
