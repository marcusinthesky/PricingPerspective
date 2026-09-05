# Structured data and scholarly discovery

## Home page graph

The collection page emits one JSON-LD graph containing:

- `WebSite`
- `Organization`
- `CollectionPage`
- `ItemList` with three paper entries
- two `Person` records
- `VideoObject` with duration, dimensions, modern media formats, poster images, transcript, and watch action

## Paper pages

Each paper page emits a graph containing `WebPage`, `ScholarlyArticle`, and a two-level `BreadcrumbList`. The article record includes:

- canonical page URL and stable `@id`
- full title, abstract, summary, publication date, revision date, and keywords
- both authors by stable `Person` identifiers
- arXiv and DOI identifiers
- the arXiv PDF as an encoded `MediaObject`
- verified paper records in `sameAs`
- collection membership and subject terms

Search-result fallbacks are intentionally excluded from `sameAs` because they are not canonical publication records.

## Blog pages

The Blog collection emits a `CollectionPage` with one `ItemList` entry per note. Each note emits `WebPage`, `BlogPosting`, and a two-level `BreadcrumbList` with its publication and revision dates, author, category, keywords, and collection membership. Every `BlogPosting` also exposes its free computer-generated WebM reading as an `AudioObject`.

## Citation metadata

Each paper route also emits Highwire/Google Scholar-style `citation_*` meta tags and Dublin Core metadata. The repository includes `public/citations.bib`, `CITATION.cff`, a sitemap, `robots.txt`, and `llms.txt`.

## Validation

After building:

```bash
bun run audit:schema
```

Also validate the deployed URLs with Google's Rich Results Test and Schema.org's validator. `ScholarlyArticle` markup improves machine readability but does not guarantee a search feature or ranking.
