# Performance budget

## Targets

- Performance: at least 0.95 in Lighthouse CI
- Accessibility: 1.00
- Best practices: 1.00; the homepage allows 0.96 for YouTube's observed DevTools cookie warning
- SEO: 1.00
- First Contentful Paint: below 1.8 s on the configured test runner
- Largest Contentful Paint: below 2.5 s
- Cumulative Layout Shift: below 0.02
- Total Blocking Time: below 250 ms

These are enforced by the pinned Lighthouse CI command in `lighthouserc.json` and the GitHub Actions workflow.

The homepage exception keeps the `inspector-issues` finding visible as a warning. Every other
scored best-practices audit remains an error at 1.00, as do the category checks on other pages.
Neither standard nor privacy-enhanced YouTube embeds eliminate this provider-controlled warning.

## Design choices supporting the budget

1. Static HTML output with no request-time rendering.
2. No client component is required by the publication pages.
3. No first-party analytics, tag manager, external font, or carousel.
4. Inline SVG replaces canvas and animation libraries.
5. Four privacy-enhanced YouTube players use native lazy-loaded iframes without an embedding SDK or autoplay.
6. Video dimensions are fixed at 16:9 to prevent layout shift, with two columns on desktop and one on mobile.
7. AVIF and WebP imagery is supplied for paper exhibits and social cards; video bytes are served by YouTube.
8. The main visual is vector markup rather than a large raster hero.
9. All below-the-fold sections use ordinary document flow and semantic elements.
10. The static export keeps deployment on GitHub Pages' CDN; host-level cache headers are not
    represented in the application source.

## Remote PageSpeed caveats

A score from PageSpeed Insights also reflects the final host, compression, cache headers, redirects, DNS/TLS setup, canonical domain, and any scripts added after handoff. Before launch:

```bash
NEXT_PUBLIC_SITE_URL=https://marcusinthesky.github.io/PricingPerspective \\
NEXT_PUBLIC_BASE_PATH=/PricingPerspective bun run build
bun run verify:export
bun run audit:schema
```

Then run PageSpeed Insights against the deployed HTTPS URL on both mobile and desktop. The embedded YouTube players can add third-party requests as they approach the viewport; account for those requests when assessing the deployed result.
