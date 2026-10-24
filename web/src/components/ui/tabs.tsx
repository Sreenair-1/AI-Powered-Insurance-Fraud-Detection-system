import * as TabsPrimitive from "@radix-ui/react-tabs";
import { forwardRef, type ComponentPropsWithoutRef, type ElementRef } from "react";
import { cn } from "@/lib/utils";

export const Tabs = TabsPrimitive.Root;
export const TabsContent = TabsPrimitive.Content;

export const TabsList = forwardRef<
  ElementRef<typeof TabsPrimitive.List>,
  ComponentPropsWithoutRef<typeof TabsPrimitive.List>
>(({ className, ...props }, ref) => (
  <TabsPrimitive.List
    ref={ref}
    className={cn("inline-flex items-center gap-1 rounded-lg border border-border bg-surface-sunken p-1", className)}
    {...props}
  />
));
TabsList.displayName = "TabsList";

export const TabsTrigger = forwardRef<
  ElementRef<typeof TabsPrimitive.Trigger>,
  ComponentPropsWithoutRef<typeof TabsPrimitive.Trigger>
>(({ className, ...props }, ref) => (
  <TabsPrimitive.Trigger
    ref={ref}
    className={cn(
      "inline-flex h-8 cursor-pointer items-center gap-1.5 rounded-md px-3 text-xs font-medium text-text-muted transition-colors duration-fast hover:text-text data-[state=active]:bg-surface data-[state=active]:text-text data-[state=active]:shadow-xs [&_svg]:size-3.5",
      className,
    )}
    {...props}
  />
));
TabsTrigger.displayName = "TabsTrigger";
