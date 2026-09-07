import { ArrowUpRight } from "lucide-react";
import Link from "next/link";

import { Badge } from "@/components/ui/badge";
import { Card } from "@/components/ui/card";
import { formatReadingTime } from "@/lib/reading-time";
import type { BlogPost } from "@/lib/types";
import { formatDate } from "@/lib/utils";

export function BlogCard({ post }: { post: BlogPost }) {
  return (
    <Card className="blog-card">
      <div className="blog-card-topline">
        <span>{post.number}</span>
        <span>{formatReadingTime(post.readingTimeMinutes)}</span>
        <time dateTime={post.datePublished}>{formatDate(post.datePublished)}</time>
      </div>
      <div>
        <p className="blog-card-category">{post.category}</p>
        <h3 className="blog-card-title">
          <Link href={`/blog/${post.slug}/`}>{post.title}</Link>
        </h3>
        <p className="blog-card-summary">{post.summary}</p>
      </div>
      <div className="badge-row">
        {post.keywords.map((keyword) => (
          <Badge key={keyword}>{keyword}</Badge>
        ))}
      </div>
      <Link className="blog-read" href={`/blog/${post.slug}/`}>
        Read the note <ArrowUpRight aria-hidden="true" size={17} />
      </Link>
    </Card>
  );
}
