import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The client always calls same-origin `/api/*`. In dev, Vite proxies those
// requests to the FastAPI server on 127.0.0.1:8000 so there is no CORS dance
// and no hard-coded backend host in the client code.
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      "/api": {
        target: "http://127.0.0.1:8000",
        changeOrigin: true,
      },
    },
  },
});
