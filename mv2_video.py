"""
🎬 Memvid Byte Video Visualizer
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Creates an MP4 video showing animated hex dump or byte heatmap
of a Memvid .mv2 file, synced to audio.

SETUP:
    pip install moviepy pillow numpy

USAGE:
    python mv2_video.py --input comments.mv2 --audio comment_banger.wav --mode hex
    python mv2_video.py --input comments.mv2 --audio comment_banger.wav --mode heatmap
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import argparse
import os
import sys
import textwrap
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from moviepy.editor import ImageSequenceClip, AudioFileClip


def read_mv2_bytes(path: str) -> bytes:
    """Read all bytes from .mv2 file."""
    with open(path, 'rb') as f:
        return f.read()


def render_hex_frame(data: bytes, offset: int, rows: int = 16, 
                     width: int = 1280, height: int = 720) -> np.ndarray:
    """Render a single frame of hex dump at given byte offset."""
    img = Image.new('RGB', (width, height), (10, 10, 20))
    draw = ImageDraw.Draw(img)
    
    try:
        font_mono = ImageFont.truetype("/System/Library/Fonts/Monaco.dfont", 16)
        font_header = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 20)
    except:
        try:
            font_mono = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 16)
            font_header = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)
        except:
            font_mono = font_header = ImageFont.load_default()
    
    # Title
    draw.text((20, 10), f"Memvid .mv2 Hex Dump - Offset 0x{offset:06x}", 
              fill=(0, 255, 200), font=font_header)
    
    # Hex dump rows
    bytes_per_row = 16
    y = 50
    
    for row in range(rows):
        row_offset = offset + row * bytes_per_row
        if row_offset >= len(data):
            break
        
        chunk = data[row_offset:row_offset + bytes_per_row]
        
        # Offset
        draw.text((20, y), f"{row_offset:06x}", fill=(100, 100, 100), font=font_mono)
        
        # Hex bytes
        for i, b in enumerate(chunk):
            x_hex = 100 + i * 28
            # Color based on byte value
            if b < 32:
                color = (150, 100, 200)  # Purple for control chars
            elif b < 128:
                color = (100, 200, 255)  # Cyan for ASCII
            else:
                color = (100, 255, 150)  # Green for high bytes
            draw.text((x_hex, y), f"{b:02x}", fill=color, font=font_mono)
        
        # ASCII
        ascii_str = ''.join(chr(b) if 32 <= b < 127 else '.' for b in chunk)
        draw.text((560, y), ascii_str, fill=(200, 200, 200), font=font_mono)
        
        y += 22
    
    # Progress bar at bottom
    progress = offset / max(1, len(data) - rows * bytes_per_row)
    bar_width = width - 40
    draw.rectangle([20, height-30, width-20, height-20], fill=(50, 50, 50), outline=(100, 100, 100))
    draw.rectangle([20, height-30, 20 + int(bar_width * progress), height-20], fill=(0, 200, 150))
    
    return np.array(img)


def render_heatmap_frame(data: bytes, offset: int, grid_width: int = 64,
                         width: int = 1280, height: int = 720) -> np.ndarray:
    """Render a scrolling heatmap frame."""
    img = Image.new('RGB', (width, height), (10, 10, 20))
    draw = ImageDraw.Draw(img)
    
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 20)
    except:
        font = ImageFont.load_default()
    
    # Title
    draw.text((20, 10), f"Memvid .mv2 Byte Heatmap - Offset 0x{offset:06x}", 
              fill=(0, 255, 200), font=font)
    
    # Calculate grid
    grid_height = (height - 100) // 8
    visible_rows = grid_height
    bytes_per_frame = grid_width * visible_rows
    
    # Draw heatmap
    pixel_size = (width - 40) // grid_width
    
    for row in range(visible_rows):
        for col in range(grid_width):
            byte_idx = offset + row * grid_width + col
            if byte_idx >= len(data):
                break
            
            b = data[byte_idx]
            x = 20 + col * pixel_size
            y = 50 + row * pixel_size
            
            # Color gradient: black -> purple -> blue -> cyan -> green -> yellow -> white
            if b == 0:
                color = (0, 0, 0)
            elif b < 32:
                color = (b * 4, 0, b * 6)
            elif b < 64:
                color = (0, (b-32) * 4, 255)
            elif b < 96:
                color = (0, 255, 255 - (b-64) * 4)
            elif b < 128:
                color = ((b-96) * 4, 255, 0)
            elif b < 192:
                color = (255, 255 - (b-128) * 2, 0)
            else:
                color = (255, 128 - (b-192), 255)
            
            draw.rectangle([x, y, x+pixel_size-1, y+pixel_size-1], fill=color)
    
    # Progress bar
    progress = offset / max(1, len(data) - bytes_per_frame)
    bar_width = width - 40
    draw.rectangle([20, height-30, width-20, height-20], fill=(50, 50, 50), outline=(100, 100, 100))
    draw.rectangle([20, height-30, 20 + int(bar_width * progress), height-20], fill=(0, 200, 150))
    
    return np.array(img)


def create_hex_video(data: bytes, audio_path: str, output_path: str, 
                     fps: int = 24, duration: float = None) -> str:
    """Create scrolling hex dump video."""
    rows_per_frame = 16
    bytes_per_row = 16
    bytes_per_frame = 8  # Scroll speed
    
    # Calculate total frames
    if duration:
        total_frames = int(duration * fps)
    else:
        total_frames = (len(data) - rows_per_frame * bytes_per_row) // bytes_per_frame
    
    frames = []
    print(f"🎬 Rendering {total_frames} hex frames...")
    
    for i in range(total_frames):
        offset = i * bytes_per_frame
        if offset >= len(data) - rows_per_frame * bytes_per_row:
            break
        
        frame = render_hex_frame(data, offset, rows_per_frame)
        frames.append(frame)
        
        if (i + 1) % 30 == 0:
            print(f"   Frame {i+1}/{total_frames}")
    
    # Create video
    print(f"\n🎞️  Encoding {len(frames)} frames...")
    clip = ImageSequenceClip(frames, fps=fps)
    
    # Add audio
    if audio_path and os.path.exists(audio_path):
        audio = AudioFileClip(audio_path)
        duration = min(clip.duration, audio.duration)
        clip = clip.subclip(0, duration)
        audio = audio.subclip(0, duration)
        clip = clip.set_audio(audio)
    
    clip.write_videofile(output_path, fps=fps, codec='libx264', 
                         audio_codec='aac', logger=None)
    print(f"✅ Video saved: {output_path}")
    return output_path


def create_heatmap_video(data: bytes, audio_path: str, output_path: str,
                        fps: int = 24, duration: float = None) -> str:
    """Create scrolling heatmap video."""
    grid_width = 64
    bytes_per_scroll = 4  # Scroll speed
    
    # Calculate frames
    visible_bytes = grid_width * 80  # approx rows visible
    if duration:
        total_frames = int(duration * fps)
    else:
        total_frames = (len(data) - visible_bytes) // bytes_per_scroll
    
    frames = []
    print(f"🎬 Rendering {total_frames} heatmap frames...")
    
    for i in range(total_frames):
        offset = i * bytes_per_scroll
        if offset >= len(data) - visible_bytes:
            break
        
        frame = render_heatmap_frame(data, offset, grid_width)
        frames.append(frame)
        
        if (i + 1) % 30 == 0:
            print(f"   Frame {i+1}/{total_frames}")
    
    # Create video
    print(f"\n🎞️  Encoding {len(frames)} frames...")
    clip = ImageSequenceClip(frames, fps=fps)
    
    # Add audio
    if audio_path and os.path.exists(audio_path):
        audio = AudioFileClip(audio_path)
        duration = min(clip.duration, audio.duration)
        clip = clip.subclip(0, duration)
        audio = audio.subclip(0, duration)
        clip = clip.set_audio(audio)
    
    clip.write_videofile(output_path, fps=fps, codec='libx264',
                         audio_codec='aac', logger=None)
    print(f"✅ Video saved: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="🎬 Create video visualization of Memvid .mv2 file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
        Video modes:
          hex      - Scrolling hex dump with audio
          heatmap  - Animated byte heatmap with audio

        Examples:
          python mv2_video.py --input comments.mv2 --audio comment_banger.wav --mode hex
          python mv2_video.py --input comments.mv2 --audio comment_banger.wav --mode heatmap --fps 30
          python mv2_video.py --input comments.mv2 --mode heatmap --duration 30
        """)
    )
    parser.add_argument("--input", required=True, help="Input Memvid .mv2 file")
    parser.add_argument("--audio", help="Audio file (WAV/MP3)")
    parser.add_argument("--mode", default="hex", choices=["hex", "heatmap"],
                        help="Video mode")
    parser.add_argument("--output", help="Output MP4 file (auto if not set)")
    parser.add_argument("--fps", type=int, default=24, help="Frames per second")
    parser.add_argument("--duration", type=float, help="Video duration in seconds")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        sys.exit(f"❌ File not found: {args.input}")

    print("━" * 50)
    print(f"   🎬 Memvid Byte Video - {args.mode.upper()} Mode")
    print("━" * 50)

    # Read data
    print(f"\n📦 Reading {args.input}...")
    data = read_mv2_bytes(args.input)
    print(f"   File size: {len(data):,} bytes")

    # Auto-generate output name
    output = args.output or f"{args.input}_{args.mode}.mp4"

    # Create video
    if args.mode == "hex":
        create_hex_video(data, args.audio, output, args.fps, args.duration)
    else:
        create_heatmap_video(data, args.audio, output, args.fps, args.duration)

    print(f"\n🎉 Done! Open: {output}")


if __name__ == "__main__":
    main()
