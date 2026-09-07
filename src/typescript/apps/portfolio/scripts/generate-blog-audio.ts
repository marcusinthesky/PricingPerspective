#!/usr/bin/env bun

import { mkdir, mkdtemp, readFile, readdir, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { basename, join, resolve } from "node:path";

import matter from "gray-matter";
import { KokoroTTS, TextSplitterStream } from "kokoro-js";

import { narrationFromMdx } from "./blog-narration";

const portfolioRoot = resolve(import.meta.dir, "..");
const contentRoot = resolve(portfolioRoot, "src/content/blog");
const audioRoot = resolve(portfolioRoot, "public/audio");
const modelId = "onnx-community/Kokoro-82M-v1.0-ONNX";
const ffmpegBinary = process.env.FFMPEG_BIN ?? "ffmpeg";
const ffprobeBinary = process.env.FFPROBE_BIN ?? "ffprobe";

const posts = await Promise.all(
  (await readdir(contentRoot, { withFileTypes: true }))
    .filter((entry) => entry.isFile() && entry.name.endsWith(".mdx"))
    .map(async (entry) => {
      const source = await readFile(join(contentRoot, entry.name), "utf8");
      const title = (matter(source).data as Record<string, unknown>).title;
      if (typeof title !== "string" || title.trim() === "") {
        throw new Error(`${entry.name}: frontmatter field 'title' is required for audio.`);
      }
      return { slug: basename(entry.name, ".mdx"), title };
    }),
);

const requestedSlug = Bun.argv
  .find((argument) => argument.startsWith("--slug="))
  ?.slice("--slug=".length);
const postsToRender = requestedSlug ? posts.filter((post) => post.slug === requestedSlug) : posts;
if (requestedSlug && postsToRender.length === 0) {
  throw new Error(`Unknown blog slug: ${requestedSlug}.`);
}

async function runFfmpeg(args: string[]): Promise<void> {
  const child = Bun.spawn([ffmpegBinary, "-y", "-hide_banner", "-loglevel", "error", ...args], {
    stdout: "inherit",
    stderr: "inherit",
  });
  const exitCode = await child.exited;
  if (exitCode !== 0) throw new Error(`ffmpeg exited with status ${exitCode}.`);
}

async function audioDuration(path: string): Promise<number> {
  const child = Bun.spawn(
    [
      ffprobeBinary,
      "-v",
      "error",
      "-show_entries",
      "format=duration",
      "-of",
      "default=noprint_wrappers=1:nokey=1",
      path,
    ],
    { stdout: "pipe", stderr: "inherit" },
  );
  const output = await new Response(child.stdout).text();
  const exitCode = await child.exited;
  if (exitCode !== 0) throw new Error(`ffprobe exited with status ${exitCode}.`);
  const duration = Number.parseFloat(output.trim());
  if (!Number.isFinite(duration)) throw new Error(`Could not read the duration of ${path}.`);
  return duration;
}

function webVttTimestamp(seconds: number): string {
  const milliseconds = Math.ceil(seconds * 1000);
  const hours = Math.floor(milliseconds / 3_600_000);
  const minutes = Math.floor((milliseconds % 3_600_000) / 60_000);
  const remainingSeconds = Math.floor((milliseconds % 60_000) / 1000);
  const remainingMilliseconds = milliseconds % 1000;
  return `${hours.toString().padStart(2, "0")}:${minutes
    .toString()
    .padStart(2, "0")}:${remainingSeconds.toString().padStart(2, "0")}.${remainingMilliseconds
    .toString()
    .padStart(3, "0")}`;
}

function concatEntry(path: string): string {
  return `file '${path.replaceAll("'", "'\\''")}'`;
}

function narrationChunks(narration: string): string[] {
  const sentenceSplitter = new TextSplitterStream();
  sentenceSplitter.push(narration);
  sentenceSplitter.close();
  const sentences = [...sentenceSplitter];
  const sentencesPerChunk = 8;
  const chunks: string[] = [];
  for (let index = 0; index < sentences.length; index += sentencesPerChunk) {
    chunks.push(sentences.slice(index, index + sentencesPerChunk).join(" "));
  }
  return chunks;
}

async function writeTranscript(post: (typeof posts)[number], narration: string): Promise<void> {
  const audioPath = join(audioRoot, `${post.slug}.webm`);
  const duration = await audioDuration(audioPath);
  const vtt = [
    "WEBVTT",
    "",
    `1`,
    `00:00:00.000 --> ${webVttTimestamp(duration)}`,
    narration,
    "",
  ].join("\n");
  await writeFile(join(audioRoot, `${post.slug}.vtt`), vtt);
}

async function renderPost(
  tts: InstanceType<typeof KokoroTTS>,
  post: (typeof posts)[number],
  temporaryRoot: string,
): Promise<void> {
  const sourcePath = join(contentRoot, `${post.slug}.mdx`);
  const source = await readFile(sourcePath, "utf8");
  const narration = narrationFromMdx(source, sourcePath);
  const chunkPaths: string[] = [];

  let chunkIndex = 0;
  for (const chunk of narrationChunks(narration)) {
    // Sequential generation keeps Kokoro's CPU and temporary storage bounded.
    // oxlint-disable-next-line no-await-in-loop
    const audio = await tts.generate(chunk);
    const chunkPath = join(temporaryRoot, `${post.slug}-${chunkIndex++}.wav`);
    // oxlint-disable-next-line no-await-in-loop
    await audio.save(chunkPath);
    chunkPaths.push(chunkPath);
  }
  if (chunkPaths.length === 0) throw new Error(`${post.slug}: Kokoro produced no audio chunks.`);

  const joinedWavPath = join(temporaryRoot, `${post.slug}-joined.wav`);
  const concatPath = join(temporaryRoot, `${post.slug}.concat.txt`);
  await writeFile(concatPath, `${chunkPaths.map(concatEntry).join("\n")}\n`);
  await runFfmpeg([
    "-f",
    "concat",
    "-safe",
    "0",
    "-i",
    concatPath,
    "-c:a",
    "pcm_s16le",
    joinedWavPath,
  ]);
  await runFfmpeg([
    "-i",
    joinedWavPath,
    "-c:a",
    "libopus",
    "-b:a",
    "96k",
    "-vbr",
    "on",
    "-application",
    "audio",
    join(audioRoot, `${post.slug}.webm`),
  ]);
  await runFfmpeg(["-i", join(audioRoot, `${post.slug}.webm`), "-f", "null", "-"]);
  await writeTranscript(post, narration);

  console.log(`Generated ${post.title}: ${narration.split(/\s+/).length} narration words.`);
}

await mkdir(audioRoot, { recursive: true });
const temporaryRoot = await mkdtemp(join(tmpdir(), "pricing-perspective-blog-audio-"));
const captionsOnly = Bun.argv.includes("--captions-only");

try {
  if (captionsOnly) {
    for (const post of postsToRender) {
      // oxlint-disable-next-line no-await-in-loop
      const source = await readFile(join(contentRoot, `${post.slug}.mdx`), "utf8");
      // oxlint-disable-next-line no-await-in-loop
      await writeTranscript(post, narrationFromMdx(source, join(contentRoot, `${post.slug}.mdx`)));
    }
  } else {
    console.log(`Loading ${modelId} on CPU (q8).`);
    const tts = await KokoroTTS.from_pretrained(modelId, {
      dtype: "q8",
      device: "cpu",
    });
    for (const post of postsToRender) {
      // Sequential rendering keeps Kokoro's CPU and temporary storage bounded.
      // oxlint-disable-next-line no-await-in-loop
      await renderPost(tts, post, temporaryRoot);
    }
  }
} finally {
  await rm(temporaryRoot, { force: true, recursive: true });
}
