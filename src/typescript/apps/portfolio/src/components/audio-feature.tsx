import { Download } from "lucide-react";

import { sitePath } from "@/lib/site";

/* oxlint-disable jsx-a11y/media-has-caption -- The supplied recording has no transcript asset. */
export function AudioFeature() {
  return (
    <div className="film-audio">
      <div className="film-audio-copy">
        <h3>Firms as news: probability distributions</h3>
        <p className="film-audio-note">NotebookLM-generated podcast · 36:11</p>
      </div>
      {/* biome-ignore lint/a11y/useMediaCaption: The supplied recording has no transcript asset. */}
      <audio
        controls
        preload="none"
        aria-label="Listen to Firms as news: probability distributions"
      >
        <source
          src={sitePath("/audio/firms-as-news-probability-distributions.webm")}
          type="audio/webm"
        />
        Your browser does not support WebM audio. Use the download link beside the player.
      </audio>
      <a
        className="film-audio-download"
        download
        href={sitePath("/audio/firms-as-news-probability-distributions.webm")}
      >
        <Download aria-hidden="true" size={16} /> Download WebM
      </a>
    </div>
  );
}
