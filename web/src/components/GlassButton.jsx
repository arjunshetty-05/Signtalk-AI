/**
 * Liquid-glass button (Apple iOS 26 "Liquid Glass" style).
 *
 * The glass look lives in `.lg-btn` in index.css; this component just wires the
 * variant class and injects the `.lg-tint` colour layer that sits under the
 * glass. The actual backdrop refraction uses the `#lg-distortion` SVG filter
 * mounted once in App.jsx.
 *
 * @param {{
 *   variant?: "brand" | "success" | "danger" | "clear",
 *   className?: string,
 *   children: React.ReactNode
 * } & React.ButtonHTMLAttributes<HTMLButtonElement>} props
 */
export default function GlassButton({
  variant = "brand",
  className = "",
  children,
  ...rest
}) {
  const variantClass =
    variant === "clear" ? "" : `lg-btn--${variant}`;
  return (
    <button type="button" className={`lg-btn ${variantClass} ${className}`} {...rest}>
      {variant !== "clear" && <span className="lg-tint" aria-hidden="true" />}
      <span className="relative z-[3] inline-flex items-center gap-2">{children}</span>
    </button>
  );
}
