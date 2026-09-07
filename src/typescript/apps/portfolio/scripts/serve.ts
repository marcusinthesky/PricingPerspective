import { extname, join, normalize, resolve } from "node:path";

const root = resolve(process.argv[2] ?? "out");
const port = Number.parseInt(process.env.PORT ?? "3000", 10);

const mime = new Map([
  [".html", "text/html; charset=utf-8"],
  [".css", "text/css; charset=utf-8"],
  [".js", "text/javascript; charset=utf-8"],
  [".json", "application/json; charset=utf-8"],
  [".xml", "application/xml; charset=utf-8"],
  [".txt", "text/plain; charset=utf-8"],
  [".svg", "image/svg+xml"],
  [".png", "image/png"],
  [".webp", "image/webp"],
  [".avif", "image/avif"],
  [".webm", "video/webm"],
  [".mp4", "video/mp4"],
  [".vtt", "text/vtt; charset=utf-8"],
]);

function safePath(pathname: string): string | undefined {
  const decoded = decodeURIComponent(pathname);
  const candidate = normalize(decoded).replace(/^([/\\])+/, "");
  const fullPath = resolve(join(root, candidate));
  return fullPath.startsWith(root) ? fullPath : undefined;
}

Bun.serve({
  port,
  async fetch(request) {
    const url = new URL(request.url);
    let filePath = safePath(url.pathname);
    if (!filePath) return new Response("Bad request", { status: 400 });

    let file = Bun.file(filePath);
    if (!(await file.exists())) {
      filePath = join(filePath, "index.html");
      file = Bun.file(filePath);
    }
    if (!(await file.exists())) return new Response("Not found", { status: 404 });

    const extension = extname(filePath);
    const immutable = /\.(?:avif|webp|png|svg|webm|mp4)$/.test(extension);
    const contentType =
      extension === ".webm" && filePath.includes("/audio/")
        ? "audio/webm"
        : (mime.get(extension) ?? "application/octet-stream");
    return new Response(file, {
      headers: {
        "Content-Type": contentType,
        "Cache-Control": immutable
          ? "public, max-age=31536000, immutable"
          : "public, max-age=0, must-revalidate",
        "X-Content-Type-Options": "nosniff",
        "Referrer-Policy": "strict-origin-when-cross-origin",
      },
    });
  },
});

console.log(`Static site: http://localhost:${port}`);
