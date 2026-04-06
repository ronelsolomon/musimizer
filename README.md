
# 🎵 YouTube Comment Musicizer

Turn any YouTube comment section into a **local AI-generated song** — 100% free, runs offline after first model download.

---

## How It Works

```
YouTube URL
    ↓
yt-dlp scrapes comments (no API key needed)
    ↓
Score & rank comments by lyric potential
    ↓
Auto-arrange into Verse / Chorus / Bridge structure
    ↓
MusicGen (Meta AI, local) generates a music track
    ↓
💾 output WAV file + lyrics TXT
```

---

## Setup

```bash
# 1. Install dependencies
pip install yt-dlp transformers torch torchaudio scipy

# Optional: YouTube API key gives more comments
pip install google-api-python-client
```

> **First run** downloads the MusicGen model (~300 MB for `small`). Cached after that.

---

## Usage

```bash
# Basic — no API key needed
python musicizer.py --url "https://youtu.be/VIDEO_ID"

# Choose genre
python musicizer.py --url "..." --genre hiphop

# Longer clip with better model
python musicizer.py --url "..." --duration 30 --model medium

# Use YouTube API key for 500+ comments
python musicizer.py --url "..." --api_key "AIza..."

# Just generate lyrics (no music, instant)
python musicizer.py --url "..." --lyrics_only

# Use Bark backend (actually sings the lyrics)
python musicizer.py --url "..." --backend bark
```

---

## Options

| Flag | Default | Description |
|------|---------|-------------|
| `--url` | *(required)* | YouTube video URL |
| `--genre` | `edm` | `edm`, `pop`, `hiphop`, `rock`, `lofi`, `trap` |
| `--duration` | `15` | Music clip length in seconds |
| `--model` | `small` | `small` (300MB), `medium` (1.5GB), `large` (3.3GB) |
| `--backend` | `musicgen` | `musicgen` (instrumental) or `bark` (singing) |
| `--api_key` | None | YouTube Data API v3 key |
| `--max_comments` | `200` | Max comments to fetch |
| `--output` | `comment_banger.wav` | Output filename |
| `--lyrics_only` | False | Skip music generation |

---

## Backends Compared

| | MusicGen | Bark |
|--|----------|------|
| **Output** | Instrumental music | Speech/singing |
| **Input** | Text music description | Actual lyrics |
| **Quality** | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| **Speed** | Fast | Slower |
| **Install** | `pip install transformers` | `pip install git+https://github.com/suno-ai/bark.git` |

---

## Hardware

| Setup | Speed for 15s clip |
|-------|--------------------|
| GPU (CUDA) | ~10–30 seconds |
| CPU only | ~2–5 minutes |
| Apple Silicon (M1/M2) | ~1–2 minutes |

---

## Output Files

```
comment_banger.wav        ← the generated music
comment_banger_lyrics.txt ← the arranged lyrics
```

Open the WAV in **Audacity**, **VLC**, or any music player.

---

## Tips

- **Best comment sections** for this: meme videos, viral clips, speedrun fails, gaming moments
- Use `--lyrics_only` first to preview what lyrics get generated before spending time on music
- Try different `--genre` flags on the same video for different vibes
- Chain with Audacity or ffmpeg to add vocal effects on top

---

## YouTube API Key (optional)

Free key from [console.cloud.google.com](https://console.cloud.google.com):
1. Create project → Enable "YouTube Data API v3"
2. Create credentials → API Key
3. Pass with `--api_key "AIza..."`

Without it, `yt-dlp` works fine for most videos.
