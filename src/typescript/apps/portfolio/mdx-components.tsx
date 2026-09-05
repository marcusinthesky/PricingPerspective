import type { MDXComponents } from "mdx/types";

import { Callout } from "@/components/callout";
import { PaperFigure } from "@/components/paper-figure";

export function useMDXComponents(components: MDXComponents): MDXComponents {
  return {
    h2: ({ children }) => <h2 className="prose-heading">{children}</h2>,
    h3: ({ children }) => <h3 className="prose-subheading">{children}</h3>,
    p: ({ children }) => <p className="prose-copy">{children}</p>,
    ul: ({ children }) => <ul className="prose-list">{children}</ul>,
    ol: ({ children }) => <ol className="prose-list prose-list-ordered">{children}</ol>,
    blockquote: ({ children }) => <blockquote className="prose-quote">{children}</blockquote>,
    Callout,
    PaperFigure,
    ...components,
  };
}
