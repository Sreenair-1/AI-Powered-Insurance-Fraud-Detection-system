import type { HTMLAttributes } from "react";
import { cn } from "@/lib/utils";

export function Skeleton({ className, ...props }: HTMLAttributes<HTMLDivElement>) {
  return (
    <div
      aria-hidden="true"
      className={cn(
        "relative overflow-hidden rounded-md bg-surface-sunken before:absolute before:inset-0 before:-translate-x-full before:animate-shimmer before:bg-gradient-to-r before:from-transparent before:via-surface/60 before:to-transparent",
        className,
      )}
      {...props}
    />
  );
}
