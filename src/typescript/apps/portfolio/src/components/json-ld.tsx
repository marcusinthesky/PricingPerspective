import type { JsonLdValue } from "@/lib/schema";

type JsonLdProps = {
  data: JsonLdValue;
};

export function JsonLd({ data }: JsonLdProps) {
  const json = JSON.stringify(data).replace(/</g, "\u003c");
  return <script type="application/ld+json" dangerouslySetInnerHTML={{ __html: json }} />;
}
