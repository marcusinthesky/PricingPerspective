import { ArrowLeft, ArrowRight } from "lucide-react";
import Link from "next/link";
import type { ReactNode } from "react";

import { Badge } from "@/components/ui/badge";
import { blogPosts } from "@/content/data";
import { formatReadingTime } from "@/lib/reading-time";
import { sitePath } from "@/lib/site";
import type { BlogPost } from "@/lib/types";
import { formatDate } from "@/lib/utils";

export function BlogShell({ post, children }: { post: BlogPost; children: ReactNode }) {
  const currentIndex = blogPosts.findIndex((item) => item.slug === post.slug);
  const previous = currentIndex > 0 ? blogPosts[currentIndex - 1] : undefined;
  const next = currentIndex < blogPosts.length - 1 ? blogPosts[currentIndex + 1] : undefined;

  return (
    <article className="blog-post-page">
      <div className="blog-post-header">
        <Link className="back-link" href="/blog/">
          <ArrowLeft aria-hidden="true" size={16} /> All blog notes
        </Link>
        <div className="blog-post-index">
          <span>Note {post.number}</span>
          <time dateTime={post.datePublished}>{formatDate(post.datePublished)}</time>
        </div>
        <p className="eyebrow blog-post-category">{post.category}</p>
        <h1>{post.title}</h1>
        <p className="blog-post-byline">
          Marcus Gawronsky · {formatReadingTime(post.readingTimeMinutes)} · {post.wordCount} words
        </p>
        <p className="blog-post-summary">{post.summary}</p>
        <div className="badge-row">
          {post.keywords.map((keyword) => (
            <Badge key={keyword}>{keyword}</Badge>
          ))}
        </div>
        <section className="blog-audio" aria-labelledby="blog-audio-title">
          <div>
            <p className="blog-audio-eyebrow" id="blog-audio-title">
              Listen to this note
            </p>
            <p className="blog-audio-note">Computer-generated reading · WebM audio</p>
          </div>
          <audio controls preload="none" aria-label={`Listen to ${post.title}`}>
            <source src={sitePath(`/audio/${post.slug}.webm`)} type="audio/webm" />
            <track
              default
              kind="captions"
              label="English transcript"
              src={sitePath(`/audio/${post.slug}.vtt`)}
              srcLang="en"
            />
            Your browser does not support WebM audio. Use the download link below.
          </audio>
          <a className="blog-audio-download" download href={sitePath(`/audio/${post.slug}.webm`)}>
            Download WebM audio
          </a>
        </section>
      </div>
      <div className="blog-post-prose">{children}</div>
      <nav className="blog-pagination" aria-label="Blog sequence">
        {previous ? (
          <Link href={`/blog/${previous.slug}/`}>
            <ArrowLeft aria-hidden="true" size={16} />
            <span>
              <small>Previous</small>
              {previous.title}
            </span>
          </Link>
        ) : (
          <span />
        )}
        {next ? (
          <Link href={`/blog/${next.slug}/`}>
            <span>
              <small>Next</small>
              {next.title}
            </span>
            <ArrowRight aria-hidden="true" size={16} />
          </Link>
        ) : (
          <span />
        )}
      </nav>
    </article>
  );
}
