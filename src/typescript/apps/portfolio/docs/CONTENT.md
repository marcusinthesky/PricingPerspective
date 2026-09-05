# Content editing guide

## Paper copy

Edit the MDX files in `src/content/papers/`. Each file contains an abstract, mathematical sequence, interpretation, and boundary statement. Standard Markdown works; the custom `<Callout title="…">` component is available.

## Cards and metadata

Edit `src/content/data.ts` for:

- titles and short titles
- dates and arXiv identifiers
- summaries and structured-data abstracts
- platform URLs
- keywords
- researcher roles, descriptions, and profile links

Paper links use `kind: "paper"` for canonical records or `kind: "search"` for title-search fallbacks; only canonical paper links enter structured-data `sameAs`. Researcher profile links use `kind: "profile"` and must point to a canonical profile.

## Home page

Edit `src/content/home.mdx` for the programme description and `src/content/authors.mdx` for the collaboration note. The short hero line and section labels live in `src/app/page.tsx` because they are also layout controls.

## Blog copy

Edit the long-form notes in `src/content/blog/`. Their titles, dates, summaries, categories, and keywords live in `src/content/data.ts`; word counts and reading times are computed from the MDX source at build time. Route wiring lives in `src/app/blog/[slug]/page.tsx`. The Blog collection is intentionally static and currently contains three entries.

## Paper exhibits

The Paper 1 MDS and Paper 2 barycentric-weight exhibits in `src/content/papers/` are rasterized web assets derived from the canonical PGF figures in `src/latex/projects/`. Re-render the corresponding PGF with `just --tempdir /tmp latex::preview` before replacing the files under `public/figures/`.

## Adding a fourth paper

1. Add a `Paper` entry to `src/content/data.ts`.
2. Create an MDX file under `src/content/papers/`.
3. Import it and add it to `contentBySlug` in `src/app/papers/[slug]/page.tsx`.
4. Add the citation to `public/citations.bib` and `CITATION.cff`.
5. Rebuild and run the structured-data audit.
