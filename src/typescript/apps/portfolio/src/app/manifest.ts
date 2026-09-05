import type { MetadataRoute } from "next";

import { sitePath } from "@/lib/site";

export const dynamic = "force-static";

export default function manifest(): MetadataRoute.Manifest {
  return {
    name: "Pricing Perspective",
    short_name: "Pricing Perspective",
    description:
      "Research papers and notes on information geometry, AI in finance, and reproducible model vintages.",
    start_url: sitePath("/"),
    display: "standalone",
    background_color: "#f7f7f2",
    theme_color: "#090909",
    icons: [
      { src: sitePath("/icons/icon-192.png"), sizes: "192x192", type: "image/png" },
      { src: sitePath("/icons/icon-512.png"), sizes: "512x512", type: "image/png" },
    ],
  };
}
