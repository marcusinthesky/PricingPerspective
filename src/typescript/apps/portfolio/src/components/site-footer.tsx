import Link from "next/link";

import { Mark } from "@/components/mark";
import { sitePath } from "@/lib/site";

export function SiteFooter() {
  return (
    <footer className="site-footer">
      <div className="site-shell footer-grid">
        <div className="footer-brand">
          <Mark />
          <p>Distribution-valued information, transport geometry, and financial dependence.</p>
        </div>
        <nav aria-label="Footer navigation">
          <Link href="/#papers">Papers</Link>
          <Link href="/#film">Video</Link>
          <Link href="/blueprint/">Proofs</Link>
          <Link href="/blog/">Blog</Link>
          <Link href="/#researchers">Researchers</Link>
          <Link href="/media-kit/">Media & partnerships</Link>
          <Link href="/privacy/">Privacy & use</Link>
          <a href={sitePath("/citations.bib")}>BibTeX</a>
          <a href={sitePath("/llms.txt")}>llms.txt</a>
        </nav>
        <p className="footer-note">
          Explore the papers, the Blog, the video, and the geometry connecting them.
          <span className="footer-copyright">© 2026 University of Cape Town</span>
        </p>
      </div>
    </footer>
  );
}
