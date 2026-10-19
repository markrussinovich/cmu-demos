import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// The FastAPI backend serves the built `dist/` in production; in dev, proxy /api to it.
export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: { dedupe: ["react", "react-dom"] },
  server: {
    port: 5175,
    proxy: { "/api": { target: "http://127.0.0.1:8082", changeOrigin: true } },
  },
  build: { outDir: "dist", emptyOutDir: true },
});
