import { AnimatePresence, motion } from "framer-motion";
import {
  Activity,
  BarChart3,
  Car,
  FileSearch,
  FilePlus2,
  HeartPulse,
  LayoutDashboard,
  Menu,
  Moon,
  Search,
  SlidersHorizontal,
  Sun,
  X,
} from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { NavLink, useLocation, useNavigate, useOutlet } from "react-router-dom";
import { useHealth } from "@/lib/api";
import { useDomain } from "@/lib/domain";
import { DOMAIN_LABEL } from "@/lib/format";
import { SAMPLE_LIMITS, useSettings, type SampleParams } from "@/lib/settings-store";
import type { Domain } from "@/lib/types";
import { cn, usePrefersReducedMotion } from "@/lib/utils";
import { Button } from "./ui/button";
import { Dialog, DialogClose, DialogContent, DialogTrigger } from "./ui/dialog";
import { Input, Label } from "./ui/input";
import { Tooltip } from "./ui/tooltip";

const NAV = [
  { to: "overview", label: "Overview", icon: LayoutDashboard },
  { to: "claims", label: "Claims investigation", icon: Search },
  { to: "forensics", label: "Document forensics", icon: FileSearch },
  { to: "assess", label: "New claim assessment", icon: FilePlus2 },
  { to: "insights", label: "Model insights", icon: BarChart3 },
] as const;

function Brand() {
  return (
    <div className="flex items-center gap-2.5">
      <svg viewBox="0 0 32 32" className="size-8 shrink-0" aria-hidden="true">
        <rect width="32" height="32" rx="7" className="fill-primary" />
        <path d="M16 6l8 3v6c0 5-3.4 9.3-8 11-4.6-1.7-8-6-8-11V9z" fill="none" className="stroke-primary-on" strokeWidth="2" />
        <circle cx="16" cy="15" r="3" className="fill-accent" />
      </svg>
      <div className="leading-tight">
        <p className="text-sm font-semibold tracking-tight text-text">InsureSight AI</p>
        <p className="text-2xs text-text-subtle">Claim risk decision support</p>
      </div>
    </div>
  );
}

function SideNav({ domain, onNavigate }: { domain: Domain; onNavigate?: () => void }) {
  return (
    <nav aria-label="Primary">
      <ul className="space-y-0.5">
        {NAV.map(({ to, label, icon: Icon }) => (
          <li key={to}>
            <NavLink
              to={`/${domain}/${to}`}
              onClick={onNavigate}
              className={({ isActive }) =>
                cn(
                  "relative flex items-center gap-2.5 rounded-md px-3 py-2 text-sm font-medium transition-colors duration-fast",
                  isActive
                    ? "bg-primary-soft text-primary before:absolute before:inset-y-1.5 before:left-0 before:w-[3px] before:rounded-full before:bg-accent"
                    : "text-text-muted hover:bg-surface-sunken hover:text-text",
                )
              }
            >
              <Icon className="size-4 shrink-0" aria-hidden="true" />
              {label}
            </NavLink>
          </li>
        ))}
      </ul>
    </nav>
  );
}

function DomainSwitcher({ domain }: { domain: Domain }) {
  const navigate = useNavigate();
  const location = useLocation();
  const go = (next: Domain) => {
    if (next === domain) return;
    // Keep the current page; a claim detail does not exist in the other domain.
    const [, , page = "overview"] = location.pathname.split("/");
    navigate(`/${next}/${page}`);
  };
  return (
    <div role="radiogroup" aria-label="Insurance domain" className="inline-flex rounded-lg border border-border bg-surface-sunken p-0.5">
      {(["health", "motor"] as Domain[]).map((d) => {
        const Icon = d === "health" ? HeartPulse : Car;
        const active = d === domain;
        return (
          <button
            key={d}
            type="button"
            role="radio"
            aria-checked={active}
            onClick={() => go(d)}
            className={cn(
              "relative inline-flex h-8 cursor-pointer items-center gap-1.5 rounded-md px-3 text-xs font-semibold transition-colors duration-fast",
              active ? "text-primary-on" : "text-text-muted hover:text-text",
            )}
          >
            {active ? (
              <motion.span layoutId="domain-pill" className="absolute inset-0 rounded-md bg-primary shadow-xs" transition={{ type: "spring", stiffness: 500, damping: 38 }} />
            ) : null}
            <Icon className="relative size-3.5" aria-hidden="true" />
            <span className="relative">{DOMAIN_LABEL[d]}</span>
          </button>
        );
      })}
    </div>
  );
}

function ApiStatus({ domain }: { domain: Domain }) {
  const { data, isError, isLoading } = useHealth();
  const slot = data?.domains.find((d) => d.domain === domain);
  const ok = !!slot?.model_loaded;
  const text = isLoading ? "Connecting…" : isError ? "API offline" : ok ? "Model ready" : "Model missing";
  const detail = isError
    ? "Start the API with `uvicorn api.main:app --port 8000`."
    : slot?.error ?? (slot ? `${slot.n_features} features · ${slot.model_dir}` : "");
  return (
    <Tooltip content={detail || text} side="bottom">
      <span
        tabIndex={0}
        className="hidden items-center gap-1.5 rounded-full border border-border px-2.5 py-1 text-xs text-text-muted sm:inline-flex"
        role="status"
      >
        <span
          className={cn(
            "size-2 rounded-full",
            isLoading ? "bg-text-subtle" : ok && !isError ? "bg-risk-low-solid" : "bg-risk-high-solid",
          )}
          aria-hidden="true"
        />
        <Activity className="size-3.5" aria-hidden="true" />
        {text}
      </span>
    </Tooltip>
  );
}

function SampleSettings() {
  const { sample, setSample, resetSample } = useSettings();
  const [draft, setDraft] = useState<SampleParams>(sample);
  return (
    <Dialog onOpenChange={(open) => open && setDraft(sample)}>
      <DialogTrigger asChild>
        <Button variant="ghost" size="icon" aria-label="Sample settings" title="Sample settings">
          <SlidersHorizontal aria-hidden="true" />
        </Button>
      </DialogTrigger>
      <DialogContent
        title="Loaded sample"
        description="The workspace scores a bounded, seeded sample of real claims, exactly like the Streamlit app. Larger samples take longer to build."
      >
        <form
          className="space-y-4"
          onSubmit={(e) => {
            e.preventDefault();
            setSample(draft);
          }}
        >
          {(Object.keys(SAMPLE_LIMITS) as (keyof SampleParams)[]).map((key) => {
            const lim = SAMPLE_LIMITS[key];
            return (
              <div key={key} className="space-y-1.5">
                <div className="flex items-center justify-between">
                  <Label htmlFor={key}>{lim.label}</Label>
                  <span className="tabular text-2xs text-text-subtle">
                    {lim.min}–{lim.max}
                  </span>
                </div>
                <div className="flex items-center gap-3">
                  <input
                    type="range"
                    aria-label={`${lim.label} slider`}
                    min={lim.min}
                    max={lim.max}
                    step={lim.step}
                    value={draft[key]}
                    onChange={(e) => setDraft({ ...draft, [key]: Number(e.target.value) })}
                    className="flex-1 accent-[rgb(var(--primary))]"
                  />
                  <Input
                    id={key}
                    type="number"
                    inputMode="numeric"
                    min={lim.min}
                    max={lim.max}
                    step={lim.step}
                    value={draft[key]}
                    onChange={(e) => setDraft({ ...draft, [key]: Number(e.target.value) })}
                    className="tabular w-24"
                  />
                </div>
              </div>
            );
          })}
          <div className="flex justify-between gap-2 pt-2">
            <Button variant="ghost" onClick={() => resetSample()}>
              Reset to defaults
            </Button>
            <DialogClose asChild>
              <Button type="submit">Apply</Button>
            </DialogClose>
          </div>
        </form>
      </DialogContent>
    </Dialog>
  );
}

export function AppShell() {
  // useOutlet() captures this route's element so the exiting page keeps
  // rendering its own content during the transition.
  const outlet = useOutlet();
  const domain = useDomain();
  const location = useLocation();
  const reduced = usePrefersReducedMotion();
  const { theme, toggleTheme } = useSettings();
  const [menuOpen, setMenuOpen] = useState(false);
  const mainRef = useRef<HTMLElement>(null);

  // Move focus to the main region on navigation (screen-reader friendly).
  useEffect(() => {
    mainRef.current?.focus({ preventScroll: true });
    window.scrollTo({ top: 0 });
    setMenuOpen(false);
  }, [location.pathname]);

  return (
    <div className="min-h-dvh">
      <a
        href="#main"
        className="sr-only z-[100] rounded-md bg-primary px-3 py-2 text-primary-on focus:not-sr-only focus:fixed focus:left-3 focus:top-3"
      >
        Skip to main content
      </a>

      <aside className="fixed inset-y-0 left-0 z-30 hidden w-[var(--sidebar-width)] flex-col border-r border-border bg-surface lg:flex">
        <div className="flex h-[var(--topbar-height)] items-center border-b border-border px-4">
          <Brand />
        </div>
        <div className="flex-1 overflow-y-auto p-3">
          <SideNav domain={domain} />
        </div>
        <div className="border-t border-border p-4 text-2xs leading-relaxed text-text-subtle">
          B.Tech project · Symbiosis Institute of Technology. Outputs support human review and never replace it.
        </div>
      </aside>

      {menuOpen ? (
        <div className="fixed inset-0 z-40 lg:hidden" role="dialog" aria-modal="true" aria-label="Navigation">
          <button
            type="button"
            className="absolute inset-0 cursor-default bg-[rgb(var(--navy-950)/0.45)]"
            aria-label="Close navigation"
            onClick={() => setMenuOpen(false)}
          />
          <div className="absolute inset-y-0 left-0 w-72 border-r border-border bg-surface p-3 shadow-lg">
            <div className="mb-3 flex items-center justify-between px-1">
              <Brand />
              <Button variant="ghost" size="icon-sm" onClick={() => setMenuOpen(false)} aria-label="Close navigation">
                <X aria-hidden="true" />
              </Button>
            </div>
            <SideNav domain={domain} onNavigate={() => setMenuOpen(false)} />
          </div>
        </div>
      ) : null}

      <div className="lg:pl-[var(--sidebar-width)]">
        <header className="sticky top-0 z-20 flex h-[var(--topbar-height)] items-center gap-3 border-b border-border bg-surface/90 px-4 backdrop-blur md:px-6">
          <Button variant="ghost" size="icon" className="lg:hidden" onClick={() => setMenuOpen(true)} aria-label="Open navigation">
            <Menu aria-hidden="true" />
          </Button>
          <div className="lg:hidden">
            <span className="text-sm font-semibold text-text">InsureSight AI</span>
          </div>
          <DomainSwitcher domain={domain} />
          <div className="ml-auto flex items-center gap-1.5">
            <ApiStatus domain={domain} />
            <SampleSettings />
            <Button
              variant="ghost"
              size="icon"
              onClick={toggleTheme}
              aria-label={theme === "dark" ? "Switch to light theme" : "Switch to dark theme"}
              title={theme === "dark" ? "Light theme" : "Dark theme"}
            >
              {theme === "dark" ? <Sun aria-hidden="true" /> : <Moon aria-hidden="true" />}
            </Button>
          </div>
        </header>

        <main id="main" ref={mainRef} tabIndex={-1} className="mx-auto max-w-[88rem] px-4 py-6 outline-none md:px-6 lg:py-8">
          <AnimatePresence mode="wait" initial={false}>
            <motion.div
              key={location.pathname}
              initial={reduced ? false : { opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              exit={reduced ? undefined : { opacity: 0, y: -4, transition: { duration: 0.12 } }}
              transition={{ duration: 0.22, ease: [0.22, 1, 0.36, 1] }}
            >
              {outlet}
            </motion.div>
          </AnimatePresence>
        </main>
      </div>
    </div>
  );
}
