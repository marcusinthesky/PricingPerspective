import { researchVideos } from "@/lib/site";

/* oxlint-disable react/iframe-missing-sandbox -- YouTube's cross-origin player uses its standard embed permissions. */
export function VideoFeature() {
  return (
    <div className="video-grid">
      {researchVideos.map((video) => (
        <figure className="video-card" id={`video-${video.id}`} key={video.id}>
          <div className="video-frame">
            <iframe
              width="560"
              height="315"
              src={video.embedUrl}
              title={video.title}
              loading="lazy"
              allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
              referrerPolicy="strict-origin-when-cross-origin"
              allowFullScreen
            />
          </div>
          <figcaption>
            <a href={video.href}>{video.title}</a>
          </figcaption>
        </figure>
      ))}
    </div>
  );
}
