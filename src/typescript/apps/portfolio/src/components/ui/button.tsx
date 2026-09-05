import type { AnchorHTMLAttributes, ButtonHTMLAttributes } from "react";

import { cn } from "@/lib/utils";

type ButtonVariant = "solid" | "outline" | "ghost";
type ButtonSize = "default" | "small" | "icon";

const classes = (variant: ButtonVariant, size: ButtonSize) =>
  cn("button", `button-${variant}`, `button-${size}`);

export type ButtonProps = ButtonHTMLAttributes<HTMLButtonElement> & {
  variant?: ButtonVariant;
  size?: ButtonSize;
};

export function Button({
  className,
  variant = "solid",
  size = "default",
  type = "button",
  ...props
}: ButtonProps) {
  return <button type={type} className={cn(classes(variant, size), className)} {...props} />;
}

export type ButtonLinkProps = AnchorHTMLAttributes<HTMLAnchorElement> & {
  variant?: ButtonVariant;
  size?: ButtonSize;
};

export function ButtonLink({
  className,
  variant = "solid",
  size = "default",
  children,
  ...props
}: ButtonLinkProps) {
  return (
    <a className={cn(classes(variant, size), className)} {...props}>
      {children}
    </a>
  );
}
