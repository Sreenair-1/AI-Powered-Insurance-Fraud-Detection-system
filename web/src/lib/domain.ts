import { useParams } from "react-router-dom";
import type { Domain } from "./types";

/** The domain from the current route (/:domain/...). The router rejects any other value. */
export function useDomain(): Domain {
  const { domain } = useParams();
  return domain === "health" ? "health" : "motor";
}
