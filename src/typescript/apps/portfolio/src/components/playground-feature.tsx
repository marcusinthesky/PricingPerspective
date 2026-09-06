/* oxlint-disable react/iframe-missing-sandbox -- The frame below carries marimo's
   documented embedding sandbox. oxlint objects to allow-scripts beside
   allow-same-origin, which is unavoidable for Pyodide and, on a cross-origin
   frame like this one, does not weaken this site's origin. See the note below. */
"use client";

import { ExternalLink, Play } from "lucide-react";
import { useState } from "react";

import { Button } from "@/components/ui/button";

const SPACE_URL = "https://marcusinthesky-pricingperspective.static.hf.space";

/**
 * The Wasserstein Playground, embedded from its Hugging Face Space.
 *
 * Deliberately click-to-load, like the podcast player's `preload="none"` beside
 * it. The notebook is a marimo export that boots CPython under WebAssembly and
 * installs numpy, scipy and matplotlib before it can draw anything: measured at
 * roughly 25-30 seconds, and it saturates a core while it works. Mounting the
 * iframe on page load would spend that on every visitor to the home page,
 * almost none of whom came for it.
 *
 * Sandboxed with the token set marimo documents for embedding. `allow-same-origin`
 * is required — Pyodide needs a real origin for its worker and storage — and is
 * safe here precisely because the frame is *cross*-origin: it keeps hf.space as
 * its origin rather than borrowing ours, and the escape that pairing allows on a
 * same-origin frame does not reach this site.
 */
export function PlaygroundFeature() {
  const [started, setStarted] = useState(false);

  return (
    <div className="film-playground">
      <div className="film-audio-copy">
        <h3>Shape the laws, watch the ceiling move</h3>
        <p className="film-audio-note">Runs in your browser · ~30s to start</p>
      </div>
      {started ? (
        <iframe
          className="film-playground-frame"
          title="Wasserstein Playground interactive notebook"
          src={SPACE_URL}
          loading="eager"
          sandbox="allow-scripts allow-same-origin allow-downloads allow-popups allow-forms"
        />
      ) : (
        <Button onClick={() => setStarted(true)} variant="outline">
          <Play aria-hidden="true" size={16} /> Launch the playground
        </Button>
      )}
      <a className="film-audio-download" href={SPACE_URL} rel="noreferrer" target="_blank">
        <ExternalLink aria-hidden="true" size={16} /> Open on Hugging Face
      </a>
    </div>
  );
}
