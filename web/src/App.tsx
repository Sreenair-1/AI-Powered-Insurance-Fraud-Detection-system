import { lazy, Suspense, type ReactNode } from "react";
import { createBrowserRouter, Link, Navigate, RouterProvider, useParams } from "react-router-dom";
import { EmptyState, SkeletonRows } from "@/components/common";
import { AppShell } from "@/components/layout";
import { Button } from "@/components/ui/button";

const Overview = lazy(() => import("@/pages/Overview"));
const Claims = lazy(() => import("@/pages/Claims"));
const ClaimDetail = lazy(() => import("@/pages/ClaimDetail"));
const Forensics = lazy(() => import("@/pages/Forensics"));
const Assess = lazy(() => import("@/pages/Assess"));
const Insights = lazy(() => import("@/pages/Insights"));

function Page({ children }: { children: ReactNode }) {
  return <Suspense fallback={<SkeletonRows rows={8} />}>{children}</Suspense>;
}

function NotFound() {
  return (
    <EmptyState title="Page not found">
      <Button asChild variant="outline" size="sm" className="mt-3">
        <Link to="/motor/overview">Go to the overview</Link>
      </Button>
    </EmptyState>
  );
}

/** Only "health" and "motor" are valid domains; anything else is a 404 page. */
function DomainGuard() {
  const { domain } = useParams();
  if (domain !== "health" && domain !== "motor") return <NotFound />;
  return <AppShell />;
}

const router = createBrowserRouter([
  { path: "/", element: <Navigate to="/motor/overview" replace /> },
  {
    path: "/:domain",
    element: <DomainGuard />,
    children: [
      { index: true, element: <Navigate to="overview" replace /> },
      { path: "overview", element: <Page><Overview /></Page> },
      { path: "claims", element: <Page><Claims /></Page> },
      { path: "claims/:claimKey", element: <Page><ClaimDetail /></Page> },
      { path: "forensics", element: <Page><Forensics /></Page> },
      { path: "assess", element: <Page><Assess /></Page> },
      { path: "insights", element: <Page><Insights /></Page> },
      { path: "*", element: <NotFound /> },
    ],
  },
  { path: "*", element: <NotFound /> },
]);

export default function App() {
  return <RouterProvider router={router} />;
}
