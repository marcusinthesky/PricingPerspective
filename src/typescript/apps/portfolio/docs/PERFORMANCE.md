# Performance budget

## Targets

- Performance: at least 0.95 in Lighthouse CI
- Accessibility: 1.00
- Best practices: 1.00
- SEO: 1.00
- First Contentful Paint: below 1.8 s on the configured test runner
- Largest Contentful Paint: below 2.5 s
- Cumulative Layout Shift: below 0.02
- Total Blocking Time: below 250 ms

These are enforced by the pinned Lighthouse CI command in `lighthouserc.json` and the GitHub Actions workflow.

## Design choices supporting the budget

1. Static HTML output with no request-time rendering.
2. No client component is required by the publication pages.
3. No analytics, tag manager, cookie banner, external font, carousel, or remote embed.
4. Inline SVG replaces canvas and animation libraries.
5. The video is not autoplayed and uses `preload="none"`.
6. Video dimensions are fixed at 16:9 to prevent layout shift.
7. AVIF and WebP poster/OG formats are supplied, with a compact MP4 fallback for video.
8. The main visual is vector markup rather than a large raster hero.
9. All below-the-fold sections use ordinary document flow and semantic elements.
10. The static export keeps deployment on GitHub Pages' CDN; host-level cache headers are not
    represented in the application source.

## Remote PageSpeed caveats

A score from PageSpeed Insights also reflects the final host, compression, cache headers, redirects, DNS/TLS setup, canonical domain, and any scripts added after handoff. Before launch:

```bash
NEXT_PUBLIC_SITE_URL=https://marcusinthesky.github.io/pricing-perspective \\
NEXT_PUBLIC_BASE_PATH=/pricing-perspective bun run build
bun run verify:export
bun run audit:schema
```

Then run PageSpeed Insights against the deployed HTTPS URL on both mobile and desktop. Avoid adding third-party embeds; link out to research platforms instead.
