import Image from "next/image";

import { sitePath } from "@/lib/site";

type PaperFigureProps = {
  alt: string;
  caption: string;
  height: number;
  src: string;
  width: number;
};

export function PaperFigure({ alt, caption, height, src, width }: PaperFigureProps) {
  const imagePath = sitePath(src);

  return (
    <figure className="paper-figure">
      <a href={imagePath} aria-label="Open full-size figure">
        <Image
          src={imagePath}
          alt={alt}
          width={width}
          height={height}
          sizes="(max-width: 64rem) 100vw, 64rem"
        />
      </a>
      <figcaption>{caption}</figcaption>
    </figure>
  );
}
