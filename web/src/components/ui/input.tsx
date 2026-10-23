import { forwardRef, type InputHTMLAttributes, type LabelHTMLAttributes, type SelectHTMLAttributes } from "react";
import { cn } from "@/lib/utils";

const field =
  "h-[var(--input-height)] w-full rounded-md border border-border-strong bg-surface px-3 text-sm text-text shadow-xs transition-colors duration-fast placeholder:text-text-subtle hover:border-text-subtle disabled:cursor-not-allowed disabled:opacity-50 aria-[invalid=true]:border-risk-high";

export const Input = forwardRef<HTMLInputElement, InputHTMLAttributes<HTMLInputElement>>(({ className, ...props }, ref) => (
  <input ref={ref} className={cn(field, className)} {...props} />
));
Input.displayName = "Input";

export const Select = forwardRef<HTMLSelectElement, SelectHTMLAttributes<HTMLSelectElement>>(
  ({ className, ...props }, ref) => <select ref={ref} className={cn(field, "cursor-pointer pr-8", className)} {...props} />,
);
Select.displayName = "Select";

export function Label({ className, ...props }: LabelHTMLAttributes<HTMLLabelElement>) {
  return <label className={cn("text-xs font-medium text-text-muted", className)} {...props} />;
}
