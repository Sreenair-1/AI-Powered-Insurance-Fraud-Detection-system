import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { Toaster } from "sonner";
import { TooltipProvider } from "@/components/ui/tooltip";
import { SettingsProvider } from "@/lib/settings";
import App from "./App";
import "./styles/globals.css";

// Apply the stored theme before first paint so token-reading charts start in
// the right palette.
try {
  if (localStorage.getItem("insuresight.theme") === "dark") document.documentElement.classList.add("dark");
} catch {
  /* storage unavailable */
}

const queryClient = new QueryClient({
  defaultOptions: {
    queries: { staleTime: 60_000, refetchOnWindowFocus: false, retry: 1 },
  },
});

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <QueryClientProvider client={queryClient}>
      <SettingsProvider>
        <TooltipProvider delayDuration={250}>
          <App />
          <Toaster position="bottom-right" richColors closeButton duration={4000} />
        </TooltipProvider>
      </SettingsProvider>
    </QueryClientProvider>
  </StrictMode>,
);
