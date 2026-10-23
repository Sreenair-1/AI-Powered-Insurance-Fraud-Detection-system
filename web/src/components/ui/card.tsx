import { forwardRef, type HTMLAttributes, type ReactNode } from "react";
import { cn } from "@/lib/utils";

export const Card = forwardRef<HTMLDivElement, HTMLAttributes<HTMLDivElement>>(({ className, ...props }, ref) => (
  <div
    ref={ref}
    className={cn("rounded-[var(--card-radius)] border border-border bg-surface shadow-xs", className)}
    {...props}
  />
));
Card.displayName = "Card";

interface CardHeaderProps extends Omit<HTMLAttributes<HTMLDivElement>, "title"> {
  title: ReactNode;
  description?: ReactNode;
  action?: ReactNode;
  icon?: ReactNode;
  as?: "h2" | "h3";
}

export function CardHeader({ title, description, action, icon, as: Heading = "h2", className, ...props }: CardHeaderProps) {
  return (
    <div className={cn("flex items-start justify-between gap-4 px-5 pt-4", className)} {...props}>
      <div className="flex min-w-0 items-start gap-2.5">
        {icon ? (
          <span className="mt-0.5 text-text-subtle [&_svg]:size-4" aria-hidden="true">
            {icon}
          </span>
        ) : null}
        <div className="min-w-0">
          <Heading className="text-sm font-semibold text-text">{title}</Heading>
          {description ? <p className="mt-0.5 text-xs text-text-muted">{description}</p> : null}
        </div>
      </div>
      {action ? <div className="shrink-0">{action}</div> : null}
    </div>
  );
}

export function CardBody({ className, ...props }: HTMLAttributes<HTMLDivElement>) {
  return <div className={cn("p-5", className)} {...props} />;
}
