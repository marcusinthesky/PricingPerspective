# Shared-field identity

The site uses the quantitative hydrographic atlas language: warm chart paper,
deep navy ink, blue contour lines, serif propositions and small monospace annotations.
Components use the semantic palette in `src/styles/site.css`.

The hero shows **a shared economic field sampled by extended firm footprints**.
Contours connect two broad regions; three elliptical measures cover different
combinations of activity. Interior stippling represents distributional support.
This is conceptual artwork, explicitly labelled schematic. It is not an empirical
map, an Ising solver, a return forecast, or evidence of dynamic propagation.
The available field paper distinguishes broad comovement from daily diffusion;
the illustration must preserve that distinction.

Motion uses one shared 36-second phase: neutral → crest → neutral → trough → neutral.
The contours, firm footprints, labels and leaders sit inside the same SVG group,
so they translate and deform together. The field moves ±8px horizontally and
±12px vertically, with a 1.5% change of aspect. Exposure opacity follows that phase.
A dot traverses a static waveform with the same period, quarter-cycle extrema and
easing; its vertical position directly indicates the field displacement. The radar
remains a fixed instrument and turns once every 72 seconds.

Firm A/B/C labels identify illustrative measures without implying real companies.
No per-frame JavaScript, SVG filters, path morphing, physics runtime or additional
dependencies. The page remains a server component. A native checkbox pauses all
motion through CSS; reduced-motion visitors see a complete static plate.

The icon abstracts the plate into three open field curves. Keep it static, use a
single ink colour, and preserve clear gaps at 16px. Do not shrink the complete
hero to produce a favicon. App icons use paper linework on a navy tile.

`scripts/generate-field-art.py` uses the standard library to sample an illustrative
two-region Gaussian field and simplify contours to subpixel accuracy. It generates
the shared TypeScript geometry, static SVG plate, SVG identity and social card.
Run it from this directory, then format the TypeScript and regenerate the PNG
icons and raster social cards using the existing ImageMagick tool. The animated
plate is `src/components/economic-field.tsx`; motion and presentation belong to
`src/styles/economic-field.css`.
