#!/usr/bin/env bash
set -euo pipefail

SOURCE=${1:-public/video/distributional-information-geometry-source.mp4}
OUT=public/video
FFMPEG_BIN=${FFMPEG_BIN:-ffmpeg}

"$FFMPEG_BIN" -y -i "$SOURCE" -map 0:v:0 -map 0:a? -c:v copy -c:a aac -b:a 128k -movflags +faststart   "$OUT/distributional-information-geometry.mp4"
"$FFMPEG_BIN" -y -i "$SOURCE" -map 0:v:0 -map 0:a? -c:v libvpx-vp9 -deadline good -cpu-used 2   -crf 34 -b:v 0 -row-mt 1 -tile-columns 2 -c:a libopus -b:a 96k   "$OUT/distributional-information-geometry.webm"
"$FFMPEG_BIN" -y -ss 64 -i "$SOURCE" -frames:v 1   -vf "format=gray,eq=contrast=1.16:brightness=-0.03" /tmp/poster.png
magick /tmp/poster.png -strip -quality 82 "$OUT/poster.webp"
magick /tmp/poster.png -strip -quality 54 "$OUT/poster.avif"
rm /tmp/poster.png
