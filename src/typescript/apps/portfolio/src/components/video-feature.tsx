import { Download, Play } from "lucide-react";

import { sitePath } from "@/lib/site";

/* oxlint-disable jsx-a11y/media-has-caption -- Voice-over captions are intentionally not shown in the player. */
export function VideoFeature() {
  return (
    <div className="video-grid">
      <div className="video-frame">
        {/* biome-ignore lint/a11y/useMediaCaption: Voice-over captions are intentionally not shown in the player. */}
        <video
          controls
          playsInline
          preload="none"
          poster={sitePath("/video/poster.webp")}
          aria-describedby="film-description"
        >
          <source
            src={sitePath("/video/distributional-information-geometry.webm")}
            type="video/webm"
          />
          <source
            src={sitePath("/video/distributional-information-geometry.mp4")}
            type="video/mp4"
          />
          Your browser does not support embedded video. Use the download links beside the video.
        </video>
      </div>
      <div className="video-copy" id="film-description">
        <p className="eyebrow">Narrated video · 05:40</p>
        <h3>A visual guide to the geometry</h3>
        <p>
          The video moves from firms as probability distributions to couplings, covariance
          envelopes, target-anchored reconstruction, coherent multi-firm dispersion, and portfolio
          risk certificates.
        </p>
        <ol className="film-chapters">
          <li>
            <span>01</span> Distribution-valued firms and optimal transport
          </li>
          <li>
            <span>02</span> Covariance restriction and transmission slack
          </li>
          <li>
            <span>03</span> Directed interaction fields and spatial closure
          </li>
          <li>
            <span>04</span> Coherent portfolio dispersion and certification
          </li>
        </ol>
        <div className="download-row">
          <a download href={sitePath("/video/distributional-information-geometry.webm")}>
            <Play aria-hidden="true" size={16} /> WebM
          </a>
          <a download href={sitePath("/video/distributional-information-geometry.mp4")}>
            <Download aria-hidden="true" size={16} /> MP4
          </a>
          <a href={sitePath("/video/transcript.txt")}>
            <Download aria-hidden="true" size={16} /> Transcript
          </a>
        </div>
      </div>
    </div>
  );
}
