import { ArrowLeft } from "lucide-react";
import Link from "next/link";

export default function NotFound() {
  return (
    <section className="not-found site-shell">
      <p className="eyebrow">404 · Unmapped coordinate</p>
      <h1>This point is outside the constellation.</h1>
      <p>The requested page is not part of the paper or Blog collections.</p>
      <Link className="button button-solid button-default" href="/">
        <ArrowLeft aria-hidden="true" size={17} /> Return home
      </Link>
    </section>
  );
}
