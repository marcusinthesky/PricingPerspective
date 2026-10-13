function attributeValue(attributes: string, name: string): string | undefined {
  const match = attributes.match(new RegExp(`\\b${name}\\s*=\\s*["']([^"']+)["']`, "i"));
  return match?.[1];
}

const spokenTexCommands: Record<string, string> = {
  mid: " given ",
  leq: " at most ",
  geq: " at least ",
  in: " in ",
  forall: " for all ",
  propto: " proportional to ",
  times: " times ",
};

// Inline LaTeX is read as plain symbol names: `\widehat R_n(h)` becomes "R hat n h".
function spokenTex(tex: string): string {
  return tex
    .replace(/\\(?:widehat|hat)\s*\{?\s*([A-Za-z])\}?/g, "$1 hat ")
    .replace(/\\([A-Za-z]+)/g, (_match, command: string) => {
      if (command in spokenTexCommands) return spokenTexCommands[command];
      return /^(?:alpha|beta|gamma|delta|epsilon|theta|lambda|mu|sigma|Sigma)$/.test(command)
        ? ` ${command} `
        : " ";
    })
    .replace(/-/g, " minus ")
    .replace(/[{}^_()[\],|]/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

function readableMarkup(source: string): string {
  return source
    .replace(/\\\$|\$([^$\n]+)\$/g, (_match, tex?: string) => (tex ? spokenTex(tex) : "$"))
    .replace(/<[^>]*>/g, " ")
    .replace(/!\[([^\]]*)\]\([^)]*\)/g, "$1")
    .replace(/\[([^\]]+)\]\([^)]*\)/g, "$1")
    .replace(/https?:\/\/\S+/g, " ")
    .replace(/&amp;/g, "and")
    .replace(/&quot;/g, '"')
    .replace(/&#39;/g, "'")
    .replace(/&ndash;/g, "–")
    .replace(/&mdash;/g, " - ")
    .replace(/[`*_~]/g, "")
    .replace(/\[DOC\]/g, "document token")
    .replace(/\bV(?=\d)/g, "V ")
    .replace(/\s+/g, " ")
    .trim();
}

function withoutFrontmatter(source: string): string {
  return source.replace(/^---\r?\n[\s\S]*?\r?\n---\r?\n?/, "");
}

function spokenSentence(source: string): string {
  const sentence = readableMarkup(source);
  if (!sentence) return "";
  return /[.!?:;]$/.test(sentence) ? sentence : `${sentence}.`;
}

function captionFromBlock(block: string, sourceName: string): string {
  const caption = block.match(
    /<(?:figcaption|caption)\b[^>]*>([\s\S]*?)<\/(?:figcaption|caption)>/i,
  );
  if (!caption) {
    throw new Error(
      `${sourceName}: captioned visual block is missing a <caption> or <figcaption>.`,
    );
  }
  return spokenSentence(caption[1]);
}

function replaceCaptionedVisuals(source: string, sourceName: string): string {
  let narration = source.replace(/<figure\b[\s\S]*?<\/figure>/gi, (block) => {
    return `\n${captionFromBlock(block, sourceName)}\n`;
  });

  narration = narration.replace(/<table\b[\s\S]*?<\/table>/gi, (block) => {
    return `\n${captionFromBlock(block, sourceName)}\n`;
  });

  narration = narration.replace(
    /<([A-Z][\w.]*)\b([^>]*?)\s*\/\s*>/g,
    (fullMatch, component: string, attributes: string) => {
      const caption = attributeValue(attributes, "caption");
      if (caption) return `\n${spokenSentence(caption)}\n`;
      if (/figure|equation|table/i.test(component)) {
        throw new Error(`${sourceName}: ${component} needs a caption for audio generation.`);
      }
      return fullMatch;
    },
  );

  narration = narration.replace(/<Callout\b([^>]*)>/gi, (_fullMatch, attributes: string) => {
    const title = attributeValue(attributes, "title");
    return title ? `\n${spokenSentence(title)}\n` : "\n";
  });

  if (/(?:^|\n)\s*\|[^\n]+\|\s*\n\s*\|\s*:?-{3,}/m.test(narration)) {
    throw new Error(
      `${sourceName}: Markdown tables need a captioned <figure> for audio generation.`,
    );
  }
  if (/!\[[^\]]*\]\([^)]*\)/.test(narration)) {
    throw new Error(
      `${sourceName}: Markdown images need a captioned <figure> for audio generation.`,
    );
  }
  if (/\$\$[\s\S]*?\$\$/.test(narration)) {
    throw new Error(
      `${sourceName}: display equations need a captioned <figure> for audio generation.`,
    );
  }

  return narration;
}

function withoutComments(source: string): string {
  return source.replace(/\{\/\*[\s\S]*?\*\/\}/g, "");
}

export function narrationFromMdx(source: string, sourceName = "blog post"): string {
  const withVisualCaptions = replaceCaptionedVisuals(
    withoutComments(withoutFrontmatter(source)),
    sourceName,
  );
  const lines = withVisualCaptions.split(/\r?\n/).map((line) => {
    const withoutHeadingMarker = line.replace(/^\s{0,3}#{1,6}\s+/, "");
    return withoutHeadingMarker.replace(/^\s*(?:[-+*]|\d+[.)])\s+/, "").trim();
  });

  return readableMarkup(lines.filter(Boolean).join("\n\n"));
}
