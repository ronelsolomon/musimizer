"""
🔬 Memvid File Visualizer
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Visualizes the internal structure of a Memvid .mv2 file
- Hex dump view
- Byte heatmap
- Frame structure diagram

USAGE:
    python mv2_visualize.py --input comments.mv2 --mode heatmap
    python mv2_visualize.py --input comments.mv2 --mode hex --limit 256
    python mv2_visualize.py --input comments.mv2 --mode structure
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import argparse
import os
import sys
import textwrap
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def read_mv2_bytes(path: str, limit: int = None) -> bytes:
    """Read raw bytes from .mv2 file."""
    with open(path, 'rb') as f:
        data = f.read(limit) if limit else f.read()
    return data


def visualize_hex(data: bytes, width: int = 16) -> str:
    """Create hex dump string."""
    lines = []
    for i in range(0, len(data), width):
        chunk = data[i:i+width]
        hex_part = ' '.join(f'{b:02x}' for b in chunk)
        ascii_part = ''.join(chr(b) if 32 <= b < 127 else '.' for b in chunk)
        lines.append(f'{i:08x}  {hex_part:<{width*3}}  {ascii_part}')
    return '\n'.join(lines)


def create_byte_heatmap(data: bytes, output_path: str, width: int = 64):
    """Create a heatmap image of byte values."""
    # Pad data to fit rectangular grid
    height = (len(data) + width - 1) // width
    padded = np.zeros(height * width, dtype=np.uint8)
    padded[:len(data)] = list(data)
    grid = padded.reshape((height, width))
    
    # Create color heatmap
    img = Image.new('RGB', (width * 4, height * 4), (0, 0, 0))
    draw = ImageDraw.Draw(img)
    
    for y in range(height):
        for x in range(width):
            byte_val = grid[y, x]
            # Color mapping: low=purple, mid=blue, high=green/yellow/white
            if byte_val < 32:
                color = (byte_val * 2, 0, byte_val * 4)  # Purple
            elif byte_val < 128:
                color = (0, byte_val, 255 - byte_val)  # Blue to cyan
            else:
                color = (byte_val - 128, 255, 0)  # Green to yellow
            
            draw.rectangle([x*4, y*4, x*4+3, y*4+3], fill=color)
    
    img.save(output_path)
    return output_path


def create_structure_diagram(mv2_path: str, output_path: str):
    """Create a diagram showing the internal structure of the .mv2 file."""
    try:
        import memvid_sdk as mv
    except ImportError:
        sys.exit("❌ pip install memvid-sdk")
    
    mem = mv.use("basic", mv2_path)
    
    # Get stats
    stats = mem.stats() if hasattr(mem, 'stats') else {}
    
    # Create diagram image
    width, height = 1200, 800
    img = Image.new('RGB', (width, height), (20, 20, 30))
    draw = ImageDraw.Draw(img)
    
    try:
        font_title = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 32)
        font_text = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 18)
        font_small = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 14)
    except:
        font_title = font_text = font_small = ImageFont.load_default()
    
    # Title
    draw.text((50, 30), f"Memvid File Structure: {os.path.basename(mv2_path)}", 
              fill=(0, 255, 200), font=font_title)
    
    # File format diagram - .mv2 structure
    y_offset = 100
    
    # Header section
    header_rect = [50, y_offset, width-50, y_offset+60]
    draw.rectangle(header_rect, fill=(0, 100, 100), outline=(0, 255, 255), width=2)
    draw.text((60, y_offset+20), "Header (4KB) - Magic, Version, Capacity", 
              fill=(255, 255, 255), font=font_text)
    
    # WAL section
    y_offset += 80
    wal_rect = [50, y_offset, width-50, y_offset+40]
    draw.rectangle(wal_rect, fill=(100, 50, 0), outline=(255, 150, 0), width=2)
    draw.text((60, y_offset+12), "Embedded WAL (1-64MB) - Crash Recovery", 
              fill=(255, 255, 255), font=font_text)
    
    # Data segments (frames)
    y_offset += 60
    
    # Get comment count
    result = mem.find("*", k=1000)
    comment_count = len(result.get('hits', []))
    
    frame_height = 30
    frame_spacing = 5
    
    draw.text((50, y_offset), f"Data Segments - {comment_count} Comments Stored:", 
              fill=(200, 200, 200), font=font_text)
    y_offset += 30
    
    # Draw frames
    colors = [(0, 150, 100), (0, 100, 150), (100, 0, 150), (150, 100, 0)]
    for i in range(min(comment_count, 20)):  # Show up to 20 frames
        color = colors[i % len(colors)]
        frame_rect = [50, y_offset, width-50, y_offset+frame_height]
        draw.rectangle(frame_rect, fill=color, outline=(255, 255, 255), width=1)
        draw.text((60, y_offset+7), f"Frame {i+1}: Comment chunk with metadata", 
                  fill=(255, 255, 255), font=font_small)
        y_offset += frame_height + frame_spacing
    
    if comment_count > 20:
        draw.text((50, y_offset), f"... and {comment_count - 20} more frames", 
                  fill=(150, 150, 150), font=font_small)
        y_offset += 40
    
    # Indexes
    y_offset += 20
    indexes = [
        ("Lex Index (Tantivy)", "Full-text search index", (100, 0, 100)),
        ("Vec Index (HNSW)", "Vector similarity search", (0, 100, 100)),
        ("Time Index", "Chronological ordering", (100, 100, 0)),
        ("TOC (Footer)", "Segment offsets", (50, 50, 50))
    ]
    
    for name, desc, color in indexes:
        rect = [50, y_offset, width-50, y_offset+35]
        draw.rectangle(rect, fill=color, outline=(255, 255, 255), width=2)
        draw.text((60, y_offset+8), f"{name} - {desc}", 
                  fill=(255, 255, 255), font=font_text)
        y_offset += 45
    
    # Footer info
    y_offset += 30
    info_text = [
        f"File: {mv2_path}",
        f"Size: {os.path.getsize(mv2_path):,} bytes",
        f"Format: MV2 (Memvid Single-File Memory)",
        f"Features: No sidecar files, portable, searchable"
    ]
    
    for line in info_text:
        draw.text((50, y_offset), line, fill=(180, 180, 180), font=font_small)
        y_offset += 20
    
    img.save(output_path)
    return output_path


def create_3d_byte_visualization(data: bytes, output_path: str, width: int = 64):
    """Create a 3D-style visualization of byte values."""
    height = (len(data) + width - 1) // width
    padded = np.zeros(height * width, dtype=np.uint8)
    padded[:len(data)] = list(data)
    grid = padded.reshape((height, width))
    
    # Create larger image with 3D effect
    img = Image.new('RGB', (width * 6 + 50, height * 6 + 50), (10, 10, 20))
    draw = ImageDraw.Draw(img)
    
    for y in range(height):
        for x in range(width):
            byte_val = grid[y, x]
            
            # 3D cube effect
            base_x = x * 6 + 25
            base_y = y * 6 + 25
            
            # Color based on byte value
            if byte_val < 32:
                r, g, b = (byte_val * 4, 0, byte_val * 6)
            elif byte_val < 128:
                r, g, b = (0, byte_val * 2, 255 - byte_val * 2)
            else:
                r, g, b = ((byte_val - 128) * 2, 255, (byte_val - 128))
            
            # Draw cube with highlight
            draw.rectangle([base_x, base_y, base_x+4, base_y+4], 
                          fill=(r, g, b), outline=(min(255, r+50), min(255, g+50), min(255, b+50)))
    
    img.save(output_path)
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="🔬 Visualize Memvid .mv2 file internal structure",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
        Visualization modes:
          hex        - Text hex dump (like hexdump -C)
          heatmap    - Color heatmap of byte values (PNG)
          structure  - Diagram of MV2 file structure (PNG)
          3d         - 3D-style byte visualization (PNG)

        Examples:
          python mv2_visualize.py --input comments.mv2 --mode hex
          python mv2_visualize.py --input comments.mv2 --mode heatmap
          python mv2_visualize.py --input comments.mv2 --mode structure
        """)
    )
    parser.add_argument("--input", required=True, help="Input Memvid .mv2 file")
    parser.add_argument("--mode", default="hex", 
                        choices=["hex", "heatmap", "structure", "3d"],
                        help="Visualization mode")
    parser.add_argument("--output", help="Output file (auto-generated if not set)")
    parser.add_argument("--limit", type=int, default=None, help="Limit bytes to read")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        sys.exit(f"❌ File not found: {args.input}")

    print("━" * 50)
    print(f"   🔬 Memvid File Visualizer - {args.mode.upper()} Mode")
    print("━" * 50)

    if args.mode == "hex":
        data = read_mv2_bytes(args.input, args.limit)
        hex_dump = visualize_hex(data)
        print(f"\n📦 File: {args.input} ({len(data):,} bytes shown)")
        print("━" * 70)
        print(hex_dump)
        print("━" * 70)
        
        if args.output:
            Path(args.output).write_text(hex_dump)
            print(f"✅ Saved to: {args.output}")

    elif args.mode == "heatmap":
        data = read_mv2_bytes(args.input, args.limit)
        output = args.output or f"{args.input}_heatmap.png"
        create_byte_heatmap(data, output)
        print(f"✅ Heatmap saved: {output}")
        print(f"   Shows {len(data):,} bytes as color gradient")

    elif args.mode == "structure":
        output = args.output or f"{args.input}_structure.png"
        create_structure_diagram(args.input, output)
        print(f"✅ Structure diagram saved: {output}")

    elif args.mode == "3d":
        data = read_mv2_bytes(args.input, args.limit)
        output = args.output or f"{args.input}_3d.png"
        create_3d_byte_visualization(data, output)
        print(f"✅ 3D visualization saved: {output}")
        print(f"   Shows {len(data):,} bytes as 3D blocks")


if __name__ == "__main__":
    main()
