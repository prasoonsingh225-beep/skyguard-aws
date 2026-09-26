#!/usr/bin/env bash
set -euo pipefail

# AERO Guard SIH 26073 video builder.
# Requires: ffmpeg. Optional narration: python + gTTS.

cd "$(dirname "$0")"
OUT="aero_guard_sih_26073.mp4"
AUDIO="narration.mp3"

if command -v python >/dev/null 2>&1 && python -c "import gtts" >/dev/null 2>&1; then
  python - <<'PY'
from gtts import gTTS
from pathlib import Path
text = Path('narration.txt').read_text(encoding='utf-8')
gTTS(text=text, lang='en', slow=False).save('narration.mp3')
print('Generated narration.mp3')
PY
fi

# Build a deterministic 16:9 slideshow from SVG storyboard frames.
cat > concat.txt <<'EOF'
file 'slides/01_title.svg'
duration 8
file 'slides/02_problem.svg'
duration 20
file 'slides/03_solution.svg'
duration 22
file 'slides/04_pipeline.svg'
duration 20
file 'slides/05_demo_data.svg'
duration 18
file 'slides/06_demo_api.svg'
duration 18
file 'slides/07_dashboard.svg'
duration 15
file 'slides/08_explainability.svg'
duration 15
file 'slides/09_edge.svg'
duration 15
file 'slides/10_evaluation.svg'
duration 15
file 'slides/11_impact.svg'
duration 14
file 'slides/12_closing.svg'
duration 10
file 'slides/12_closing.svg'
EOF

ffmpeg -y -f concat -safe 0 -i concat.txt -vf "scale=1920:1080,format=yuv420p" -r 30 -c:v libx264 -pix_fmt yuv420p slideshow.mp4

if [ -f "$AUDIO" ]; then
  ffmpeg -y -i slideshow.mp4 -i "$AUDIO" -map 0:v -map 1:a -c:v copy -c:a aac -b:a 192k -shortest "$OUT"
else
  ffmpeg -y -i slideshow.mp4 -f lavfi -i anullsrc=channel_layout=stereo:sample_rate=44100 -map 0:v -map 1:a -c:v copy -c:a aac -shortest "$OUT"
fi

echo "Created: $(pwd)/$OUT"
