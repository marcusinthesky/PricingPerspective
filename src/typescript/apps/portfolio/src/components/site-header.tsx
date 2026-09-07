import Link from "next/link";

import { Mark } from "@/components/mark";

const navigation = [
  { href: "/#papers", label: "Papers" },
  { href: "/#film", label: "Video" },
  { href: "/blueprint/", label: "Proofs" },
  { href: "/blog/", label: "Blog" },
  { href: "/#researchers", label: "Researchers" },
  { href: "/media-kit/", label: "Partnerships" },
] as const;

export function SiteHeader() {
  return (
    <header className="site-header">
      <div className="site-shell header-inner">
        <Link className="brand" href="/" aria-label="Pricing Perspective home">
          <Mark />
          <span>Pricing Perspective</span>
        </Link>
        <nav aria-label="Primary navigation">
          <ul className="nav-list">
            {navigation.map((item) => (
              <li
                key={item.href}
                className={
                  item.label === "Video" || item.label === "Proofs" ? "nav-hide-small" : undefined
                }
              >
                <Link href={item.href}>{item.label}</Link>
              </li>
            ))}
          </ul>
        </nav>
      </div>
    </header>
  );
}
