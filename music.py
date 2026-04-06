"""
🎵 YouTube Comment Musicizer
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Scrapes YouTube comments → picks the funniest/rhythmic ones
→ turns them into song lyrics → generates music with MusicGen (local, free)

SETUP (run once):
    pip install yt-dlp google-api-python-client transformers torch torchaudio scipy

USAGE:
    python music.py --url "https://youtube.com/watch?v=XXXX"
    python music.py --url "..." --api_key "YOUR_YT_API_KEY"  # for more comments
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import argparse
import json
import os
import random
import re
import subprocess
import sys
import textwrap
import time
from pathlib import Path


# ──────────────────────────────────────────────
# 1. COMMENT SCRAPING
# ──────────────────────────────────────────────

def scrape_comments_ytdlp(url: str, max_comments: int = 200) -> list[dict]:
    """
    Scrape comments using yt-dlp (no API key needed).
    Returns list of {"text": ..., "likes": ...}
    """
    print("📥 Scraping comments with yt-dlp (no API key needed)...")
    try:
        result = subprocess.run(
            [
                "yt-dlp",
                "--write-comments",
                "--skip-download",
                "--no-playlist",
                "-o", "/tmp/yt_comments_%(id)s",
                "--print-json",
                url,
            ],
            capture_output=True, text=True, timeout=120
        )
        # Parse the JSON info dump
        info = json.loads(result.stdout.strip().splitlines()[-1])
        raw_comments = info.get("comments", []) or []
        comments = [
            {"text": c.get("text", ""), "likes": c.get("like_count", 0) or 0}
            for c in raw_comments
            if c.get("text", "").strip()
        ][:max_comments]
        print(f"✅ Got {len(comments)} comments via yt-dlp")
        return comments
    except Exception as e:
        print(f"⚠️  yt-dlp failed: {e}")
        return []


def scrape_comments_api(url: str, api_key: str, max_comments: int = 200) -> list[dict]:
    """
    Scrape comments using YouTube Data API v3 (needs API key, gets more comments).
    """
    try:
        from googleapiclient.discovery import build
    except ImportError:
        print("⚠️  google-api-python-client not installed. Run: pip install google-api-python-client")
        return []

    # Extract video ID
    match = re.search(r"(?:v=|youtu\.be/)([A-Za-z0-9_-]{11})", url)
    if not match:
        print("❌ Could not extract video ID from URL")
        return []
    video_id = match.group(1)

    print(f"📥 Scraping comments via YouTube API for video: {video_id}")
    youtube = build("youtube", "v3", developerKey=api_key)
    comments = []
    next_page = None

    while len(comments) < max_comments:
        try:
            resp = youtube.commentThreads().list(
                part="snippet",
                videoId=video_id,
                maxResults=min(100, max_comments - len(comments)),
                pageToken=next_page,
                textFormat="plainText",
                order="relevance",
            ).execute()
        except Exception as e:
            print(f"⚠️  API error: {e}")
            break

        for item in resp.get("items", []):
            snippet = item["snippet"]["topLevelComment"]["snippet"]
            comments.append({
                "text": snippet.get("textDisplay", ""),
                "likes": snippet.get("likeCount", 0),
            })

        next_page = resp.get("nextPageToken")
        if not next_page:
            break

    print(f"✅ Got {len(comments)} comments via API")
    return comments


# ──────────────────────────────────────────────
# 2. COMMENT SELECTION & LYRIC CRAFTING
# ──────────────────────────────────────────────

FILLER_WORDS = {"the", "a", "an", "is", "was", "are", "were", "i", "you",
                "he", "she", "it", "we", "they", "this", "that", "and",
                "but", "or", "so", "of", "to", "in", "on", "at", "for"}

def score_comment(comment: dict) -> float:
    """
    Score a comment for lyric potential:
    - Short is better (easier to sing)
    - Punctuation/rhythm markers help
    - Likes = crowd approval
    - Avoid URLs and spam
    """
    text = comment["text"].strip()
    likes = comment.get("likes", 0)

    if not text or len(text) < 5:
        return 0
    if "http" in text or len(text) > 280:
        return 0
    if text.count("\n") > 3:  # skip wall-of-text comments
        return 0

    words = text.split()
    word_count = len(words)
    unique_ratio = len(set(w.lower() for w in words)) / max(word_count, 1)
    punctuation_score = text.count("!") * 2 + text.count("?") + text.count("~")
    caps_score = sum(1 for w in words if w.isupper() and len(w) > 1)
    repeat_score = sum(1 for w in words if words.count(w) > 1) * 0.5

    # Prefer 4–20 word comments (singable length)
    length_score = 10 - abs(word_count - 10)

    score = (
        length_score * 2
        + unique_ratio * 5
        + punctuation_score
        + caps_score
        + repeat_score
        + min(likes / 10, 20)   # cap likes influence
    )
    return score


def pick_best_comments(comments: list[dict], n: int = 12) -> list[str]:
    """Sort by score, deduplicate, pick top-n."""
    scored = [(score_comment(c), c["text"].strip()) for c in comments]
    scored = [(s, t) for s, t in scored if s > 0]
    scored.sort(reverse=True)

    seen = set()
    selected = []
    for _, text in scored:
        key = text.lower()[:40]
        if key not in seen:
            seen.add(key)
            selected.append(text)
        if len(selected) >= n:
            break

    return selected


def generate_lyrics_with_ollama(comments: list[dict], title: str = "Internet Banger", ollama_model: str = "llama3.2", ollama_url: str = "http://localhost:11434") -> str:
    """
    Use Ollama to generate creative lyrics from all comments.
    Sends all comment texts to a local LLM and asks it to craft song lyrics.
    """
    # Prepare all comment texts
    all_texts = [c["text"].strip() for c in comments if c["text"].strip()]
    
    # Build the prompt
    comments_block = "\n".join(f"- {t}" for t in all_texts[:150])  # Limit to avoid token limits
    if len(all_texts) > 150:
        comments_block += f"\n... and {len(all_texts) - 150} more comments"
    
    prompt = f"""You are a creative songwriter. Create song lyrics using these YouTube comments as inspiration.

Comments:
{comments_block}

Create song lyrics with:
- [Intro] - 2-4 lines
- [Verse 1] - 4-6 lines  
- [Chorus] - 4 lines (catchy, repeatable)
- [Verse 2] - 4-6 lines
- [Chorus] - repeat
- [Bridge] - 2-4 lines (optional)
- [Outro] - 1-2 lines

Rules:
- Mix, rephrase, and combine the comment ideas creatively
- Make it flow like a real song
- Keep lines short and singable
- Output ONLY the lyrics with section labels, no explanations

Song title: {title}
"""

    print(f"\n🤖 Sending {len(all_texts)} comments to Ollama ({ollama_model}) for lyric generation...")
    
    try:
        import requests
        response = requests.post(
            f"{ollama_url}/api/generate",
            json={
                "model": ollama_model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.8,
                    "num_predict": 800,
                }
            },
            timeout=120
        )
        response.raise_for_status()
        result = response.json()
        lyrics = result.get("response", "").strip()
        
        # Clean up the response - ensure it has proper formatting
        if not lyrics.startswith("🎵"):
            lyrics = f"🎵 {title}\n" + ("━" * 40) + "\n\n" + lyrics
            
        print(f"✅ Ollama generated lyrics ({len(lyrics)} chars)")
        return lyrics
        
    except ImportError:
        print("❌ requests not installed. Run: pip install requests")
        raise
    except requests.ConnectionError:
        print(f"❌ Cannot connect to Ollama at {ollama_url}")
        print("   Make sure Ollama is running: ollama serve")
        raise
    except Exception as e:
        print(f"⚠️  Ollama generation failed: {e}")
        raise


def build_lyrics(comments: list[str], title: str = "Internet Banger") -> str:
    """
    Arrange comments into verse/chorus/bridge structure.
    Repeats the most liked comment as a "chorus hook".
    """
    if not comments:
        return "No comments found. Check your URL."

    random.shuffle(comments)

    chorus_hook = comments[0]
    verses = comments[1:]

    def chunk(lst, size):
        for i in range(0, len(lst), size):
            yield lst[i:i + size]

    lyric_blocks = [f"[Intro]\n{chorus_hook}\n"]

    verse_chunks = list(chunk(verses[:8], 4))
    for i, chunk_lines in enumerate(verse_chunks, 1):
        lyric_blocks.append(f"[Verse {i}]\n" + "\n".join(chunk_lines))
        lyric_blocks.append(f"[Chorus]\n{chorus_hook}\n{chorus_hook}")

    if len(verses) > 8:
        bridge_lines = verses[8:11]
        lyric_blocks.append(f"[Bridge]\n" + "\n".join(bridge_lines))
        lyric_blocks.append(f"[Chorus]\n{chorus_hook}\n{chorus_hook}")

    lyric_blocks.append(f"[Outro]\n{chorus_hook}...")

    full_lyrics = f"🎵 {title}\n" + ("━" * 40) + "\n\n"
    full_lyrics += "\n\n".join(lyric_blocks)
    return full_lyrics


# ──────────────────────────────────────────────
# 3. MUSIC GENERATION  (local, free, offline)
# ──────────────────────────────────────────────

def build_music_prompt(lyrics: str, genre: str = "edm") -> str:
    """
    MusicGen takes a TEXT DESCRIPTION of music, not lyrics.
    We craft a prompt from the genre + tone of comments.
    """
    genre_templates = {
        "edm":     "upbeat EDM track with heavy 808 bass, synth drops, energetic build-ups and a catchy hook, 128 BPM",
        "pop":     "catchy pop song with bright piano chords, clapping percussion, and an infectious chorus melody",
        "hiphop":  "lo-fi hip-hop beat with boom-bap drums, sampled jazz chords, and a smooth melodic hook",
        "rock":    "driving rock track with distorted electric guitar riffs, punchy drums, and a stadium-ready chorus",
        "lofi":    "chill lo-fi beat with vinyl crackle, mellow chords, soft drums, and a dreamy atmosphere",
        "trap":    "dark trap beat with 808 slides, hi-hat rolls, cinematic strings, and hard-hitting drops",
    }
    base = genre_templates.get(genre, genre_templates["edm"])
    return f"{base}, high quality, professional production, radio-ready"


def generate_music_musicgen(
    prompt: str,
    output_path: str = "output_music.wav",
    duration: int = 15,
    model_size: str = "small",
) -> str:
    """
    Generate music locally with Meta's MusicGen via HuggingFace Transformers.

    Models (auto-downloaded ~first run):
      - facebook/musicgen-small   (~300 MB, fast, good quality)
      - facebook/musicgen-medium  (~1.5 GB, better quality)
      - facebook/musicgen-large   (~3.3 GB, best quality)

    Requires: pip install transformers torch torchaudio scipy
    """
    print("\n🎼 Loading MusicGen model (downloads once, ~300MB for small)...")
    print(f"   Prompt: {prompt[:80]}...")
    print(f"   Duration: {duration}s | Model: musicgen-{model_size}")

    try:
        from transformers import AutoProcessor, MusicgenForConditionalGeneration
        import torch
        import scipy.io.wavfile
        import numpy as np
    except ImportError as e:
        print(f"\n❌ Missing dependency: {e}")
        print("Install with: pip install transformers torch torchaudio scipy")
        sys.exit(1)

    model_id = f"facebook/musicgen-{model_size}"

    # Load model
    processor = AutoProcessor.from_pretrained(model_id)
    model = MusicgenForConditionalGeneration.from_pretrained(model_id)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    print(f"   Running on: {device.upper()}")

    # Tokenize prompt
    inputs = processor(
        text=[prompt],
        padding=True,
        return_tensors="pt",
    ).to(device)

    # Generate
    sampling_rate = model.config.audio_encoder.sampling_rate
    frame_rate = getattr(model.config.audio_encoder, 'frame_rate', 50)  # default 50fps
    max_new_tokens = int(duration * sampling_rate / frame_rate)
    
    # Cap to safe limit to avoid position embedding overflow
    max_safe_tokens = 1500  # MusicGen can handle ~30s max
    if max_new_tokens > max_safe_tokens:
        print(f"   ⚠️  Capping tokens: {max_new_tokens} → {max_safe_tokens} (max context)")
        max_new_tokens = max_safe_tokens
    
    print(f"   Tokens to generate: {max_new_tokens}")

    print("🎹 Generating music... (this takes 30s–3min depending on hardware)")
    t0 = time.time()
    with torch.no_grad():
        audio_values = model.generate(**inputs, max_new_tokens=max_new_tokens, guidance_scale=1.0)
    elapsed = time.time() - t0
    print(f"✅ Generated in {elapsed:.1f}s")

    # Save to WAV
    audio_np = audio_values[0, 0].cpu().numpy()
    audio_np = (audio_np * 32767).astype("int16")
    scipy.io.wavfile.write(output_path, sampling_rate, audio_np)
    print(f"💾 Saved: {output_path}")
    return output_path


def generate_music_bark(
    lyrics_snippet: str,
    output_path: str = "output_music.wav",
) -> str:
    """
    Alternative: Use Suno's Bark for text-to-speech/singing.
    Less musical but can actually 'sing' lyrics.
    pip install git+https://github.com/suno-ai/bark.git
    """
    try:
        from bark import generate_audio, SAMPLE_RATE
        from bark.preload import preload_models
        import scipy.io.wavfile
        import numpy as np
    except ImportError:
        print("❌ Bark not installed. Run: pip install git+https://github.com/suno-ai/bark.git")
        sys.exit(1)

    print("🌳 Loading Bark model...")
    preload_models()

    # Bark uses ♪ markers for singing
    sing_text = "♪ " + lyrics_snippet[:200].replace("\n", " ♪ ") + " ♪"
    print(f"🎤 Generating: {sing_text[:80]}...")

    audio = generate_audio(sing_text)
    scipy.io.wavfile.write(output_path, SAMPLE_RATE, audio)
    print(f"💾 Saved: {output_path}")
    return output_path


# ──────────────────────────────────────────────
# 4. MAIN PIPELINE
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="🎵 YouTube Comment Musicizer — turn comment sections into bangers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
        Examples:
          python musicizer.py --url "https://youtu.be/dQw4w9WgXcQ"
          python musicizer.py --url "..." --genre hiphop --duration 20
          python musicizer.py --url "..." --api_key "AIza..." --model medium
          python musicizer.py --url "..." --backend bark
        """)
    )
    parser.add_argument("--url",      required=True,  help="YouTube video URL")
    parser.add_argument("--api_key",  default=None,   help="YouTube Data API v3 key (optional, gets more comments)")
    parser.add_argument("--genre",    default="edm",  choices=["edm","pop","hiphop","rock","lofi","trap"],
                        help="Music genre (default: edm)")
    parser.add_argument("--duration", default=15,     type=int,  help="Music clip length in seconds (default: 15)")
    parser.add_argument("--model",    default="small", choices=["small","medium","large"],
                        help="MusicGen model size (default: small)")
    parser.add_argument("--backend",  default="musicgen", choices=["musicgen","bark"],
                        help="AI backend to use (default: musicgen)")
    parser.add_argument("--ollama",   action="store_true", help="Use Ollama to generate lyrics from ALL comments (requires Ollama running)")
    parser.add_argument("--ollama_model", default="llama3.1:8b", help="Ollama model to use (default: llama3.1:8b)")
    parser.add_argument("--ollama_url", default="http://localhost:11434", help="Ollama API URL (default: http://localhost:11434)")
    parser.add_argument("--max_comments", default=200, type=int, help="Max comments to fetch")
    parser.add_argument("--output",   default="comment_banger.wav", help="Output WAV file name")
    parser.add_argument("--lyrics_only", action="store_true", help="Only generate lyrics, skip music gen")
    args = parser.parse_args()

    print("\n" + "━"*50)
    print("   🎵 YouTube Comment Musicizer")
    print("━"*50 + "\n")

    # Step 1: Scrape comments
    if args.api_key:
        comments = scrape_comments_api(args.url, args.api_key, args.max_comments)
    else:
        comments = scrape_comments_ytdlp(args.url, args.max_comments)

    if not comments:
        print("❌ No comments found. Try --api_key or check your URL.")
        sys.exit(1)

    # Step 2: Generate lyrics (using Ollama or template approach)
    if args.ollama:
        print(f"\n🧠 Using Ollama to craft lyrics from all {len(comments)} comments...")
        try:
            lyrics = generate_lyrics_with_ollama(
                comments, 
                title="Internet Banger",
                ollama_model=args.ollama_model,
                ollama_url=args.ollama_url
            )
        except Exception:
            print("\n⚠️  Falling back to template-based lyric generation...")
            best = pick_best_comments(comments, n=12)
            lyrics = build_lyrics(best, title="Internet Banger")
    else:
        print(f"\n🧠 Selecting best comments from {len(comments)} total...")
        best = pick_best_comments(comments, n=12)
        print(f"✅ Selected {len(best)} comments for lyrics\n")
        lyrics = build_lyrics(best, title="Internet Banger")
    print(lyrics)
    print()

    # Save lyrics
    lyrics_path = args.output.replace(".wav", "_lyrics.txt")
    Path(lyrics_path).write_text(lyrics)
    print(f"📝 Lyrics saved: {lyrics_path}")

    if args.lyrics_only:
        print("\n✅ Done (lyrics only mode)")
        return

    # Step 3: Generate music
    music_prompt = build_music_prompt(lyrics, genre=args.genre)
    print(f"\n🎛️  Music prompt: {music_prompt}")

    if args.backend == "musicgen":
        output_file = generate_music_musicgen(
            prompt=music_prompt,
            output_path=args.output,
            duration=args.duration,
            model_size=args.model,
        )
    else:
        # Bark sings the first chorus
        chorus_text = best[0] if best else "Hello world"
        output_file = generate_music_bark(chorus_text, output_path=args.output)

    print(f"\n🎉 All done!")
    print(f"   🎵 Music: {output_file}")
    print(f"   📝 Lyrics: {lyrics_path}")
    print(f"\nTip: Open the WAV in Audacity or any player to listen!")


if __name__ == "__main__":
    main()