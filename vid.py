"""
🎬 3D Audio Visualizer Video Generator
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Reads a WAV file → renders a 3D animated visualizer → exports MP4

SETUP (run once):
    pip install numpy scipy matplotlib moviepy pillow tqdm

USAGE:
    python visualizer.py --input comment_banger.wav
    python visualizer.py --input comment_banger.wav --style galaxy
    python visualizer.py --input comment_banger.wav --style bars3d --fps 30
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import argparse
import os
import sys
import textwrap
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np

# ──────────────────────────────────────────────
# AUDIO ANALYSIS
# ──────────────────────────────────────────────

def load_audio(path: str):
    """Load WAV, return (samples float32, sample_rate)."""
    try:
        import scipy.io.wavfile as wf
    except ImportError:
        sys.exit("❌ pip install scipy")

    rate, data = wf.read(path)
    if data.ndim > 1:
        data = data.mean(axis=1)           # stereo → mono
    data = data.astype(np.float32)
    data /= np.abs(data).max() + 1e-9     # normalise
    return data, rate


def get_fft_frames(audio: np.ndarray, rate: int, fps: int, n_bins: int = 64):
    """
    Slice audio into per-frame FFT spectra.
    Returns array shape (n_frames, n_bins), values 0-1.
    """
    hop = rate // fps
    frames = []
    for start in range(0, len(audio) - hop, hop):
        chunk = audio[start: start + hop]
        win   = np.hanning(len(chunk))
        spec  = np.abs(np.fft.rfft(chunk * win))
        # log-scale & bin down to n_bins
        spec  = np.log1p(spec)
        # resample to n_bins via strided mean
        ratio = len(spec) // n_bins
        if ratio > 1:
            spec = spec[:ratio * n_bins].reshape(n_bins, ratio).mean(axis=1)
        else:
            spec = np.interp(np.linspace(0, len(spec)-1, n_bins),
                             np.arange(len(spec)), spec)
        spec /= (spec.max() + 1e-9)
        frames.append(spec.astype(np.float32))
    return np.array(frames)


def get_rms_frames(audio: np.ndarray, rate: int, fps: int) -> np.ndarray:
    """Per-frame RMS energy, 0-1."""
    hop = rate // fps
    rms = []
    for start in range(0, len(audio) - hop, hop):
        chunk = audio[start: start + hop]
        rms.append(float(np.sqrt(np.mean(chunk**2))))
    arr = np.array(rms, dtype=np.float32)
    arr /= arr.max() + 1e-9
    return arr


# ──────────────────────────────────────────────
# STYLES
# ──────────────────────────────────────────────

def render_frame_bars3d(fft: np.ndarray, rms: float, frame_idx: int,
                        width: int, height: int) -> np.ndarray:
    """
    Classic 3D bar equalizer with perspective grid floor.
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D   # noqa: F401

    fig = plt.figure(figsize=(width/100, height/100), dpi=100, facecolor='black')
    ax  = fig.add_subplot(111, projection='3d', facecolor='black')

    n    = len(fft)
    x    = np.arange(n)
    z    = np.zeros(n)
    dx   = 0.7
    dy   = fft

    # colour: cyan → magenta gradient by frequency
    colors = [
        (0.0 + 0.9 * i/n,
         0.8 - 0.6 * fft[i],
         1.0 - 0.4 * i/n,
         0.9)
        for i in range(n)
    ]

    ax.bar3d(x - dx/2, z, z, dx, dy, dy, color=colors, shade=True, zsort='max')

    # Grid floor
    ax.set_zlim(0, 1.5)
    ax.set_xlim(-1, n)
    ax.set_ylim(-0.1, 1.2)

    # Rotate slowly
    ax.view_init(elev=25, azim=frame_idx * 0.4 % 360)

    ax.set_axis_off()

    # Glow title
    ax.text2D(0.5, 0.95, "◈ COMMENT BANGER ◈",
              transform=ax.transAxes,
              color='#00ffcc', fontsize=12, ha='center',
              fontfamily='monospace', alpha=0.9)

    fig.tight_layout(pad=0)
    img = _fig_to_array(fig)
    plt.close(fig)
    return img


def render_frame_galaxy(fft: np.ndarray, rms: float, frame_idx: int,
                        width: int, height: int) -> np.ndarray:
    """
    Spinning galaxy / radial waveform in 3D.
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    fig = plt.figure(figsize=(width/100, height/100), dpi=100, facecolor='black')
    ax  = fig.add_subplot(111, projection='3d', facecolor='black')

    n      = len(fft)
    t      = np.linspace(0, 4 * np.pi, n)
    r      = 1.0 + fft * 0.8
    angle  = frame_idx * 0.03

    # Spiral arms
    for arm in range(3):
        offset = (2 * np.pi / 3) * arm + angle
        xs = r * np.cos(t + offset)
        ys = r * np.sin(t + offset)
        zs = fft * np.sin(t * 2) * 0.5

        lw   = 1.0 + rms * 2
        alpha = 0.7 + rms * 0.3
        colors_arm = ['#00ffcc', '#ff00aa', '#ffaa00']
        ax.plot(xs, ys, zs, color=colors_arm[arm], linewidth=lw, alpha=alpha)

        # Scatter particles
        idx = np.random.choice(n, size=n//6, replace=False)
        ax.scatter(xs[idx], ys[idx], zs[idx],
                   c=colors_arm[arm], s=rms * 20 + 2, alpha=0.5)

    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-2.5, 2.5)
    ax.set_zlim(-1, 1)
    ax.view_init(elev=30 + rms * 15, azim=frame_idx * 0.6 % 360)
    ax.set_axis_off()

    ax.text2D(0.5, 0.93, "◈ COMMENT BANGER ◈",
              transform=ax.transAxes,
              color='#00ffcc', fontsize=11, ha='center',
              fontfamily='monospace')

    fig.tight_layout(pad=0)
    img = _fig_to_array(fig)
    plt.close(fig)
    return img


def render_frame_tunnel(fft: np.ndarray, rms: float, frame_idx: int,
                        width: int, height: int) -> np.ndarray:
    """
    Psychedelic 3D tunnel / wormhole that pulses with bass.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(width/100, height/100), dpi=100,
                           facecolor='black')
    ax.set_facecolor('black')
    ax.set_aspect('equal')

    cx, cy = 0.5, 0.5
    n_rings = 20
    n_pts   = 180

    for ring in range(n_rings, 0, -1):
        t        = frame_idx * 0.05 + ring * 0.15
        radius   = ring / n_rings * 0.5

        # Modulate shape by FFT bins mapped to ring
        bin_idx  = int((ring / n_rings) * (len(fft) - 1))
        distort  = fft[bin_idx] * 0.08

        angles   = np.linspace(0, 2*np.pi, n_pts)
        xs = cx + (radius + distort * np.sin(angles * 7 + t)) * np.cos(angles)
        ys = cy + (radius + distort * np.cos(angles * 5 + t)) * np.sin(angles)

        # Colour shifts with ring depth
        hue_shift = (ring / n_rings + frame_idx * 0.01) % 1.0
        r = 0.3 + 0.7 * abs(np.sin(hue_shift * np.pi * 2))
        g = 0.1 + 0.5 * abs(np.sin(hue_shift * np.pi * 2 + 2.1))
        b = 0.8 - 0.4 * abs(np.sin(hue_shift * np.pi * 2 + 4.2))
        alpha = 0.3 + 0.7 * (ring / n_rings)
        lw    = 0.5 + rms * 3 * (ring / n_rings)

        ax.plot(xs, ys, color=(r, g, b, alpha), linewidth=lw)

    # Flash on beat
    if rms > 0.6:
        flash = plt.Circle((cx, cy), 0.04, color='white', alpha=rms * 0.8)
        ax.add_patch(flash)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    ax.text(0.5, 0.96, "◈ COMMENT BANGER ◈",
            transform=ax.transAxes,
            color='#00ffcc', fontsize=11, ha='center',
            fontfamily='monospace')

    fig.tight_layout(pad=0)
    img = _fig_to_array(fig)
    plt.close(fig)
    return img


def render_frame_terrain(fft: np.ndarray, rms: float, frame_idx: int,
                         width: int, height: int) -> np.ndarray:
    """
    Audio-reactive 3D terrain / mountain range.
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    fig = plt.figure(figsize=(width/100, height/100), dpi=100, facecolor='black')
    ax  = fig.add_subplot(111, projection='3d', facecolor='black')

    n    = len(fft)
    cols = n
    rows = 32
    X, Y = np.meshgrid(np.linspace(0, 1, cols), np.linspace(0, 1, rows))
    Z    = np.zeros_like(X)

    # Build terrain: each row is slightly time-shifted FFT
    for row in range(rows):
        shift = (frame_idx + row * 2) % cols
        rolled = np.roll(fft, shift)
        Z[row, :] = rolled * (1.0 - row / rows * 0.6)

    # Neon colour surface
    facecolors = plt.cm.plasma((Z - Z.min()) / (Z.max() - Z.min() + 1e-9))

    ax.plot_surface(X, Y, Z, facecolors=facecolors,
                    rstride=1, cstride=1, linewidth=0, antialiased=True, alpha=0.9)

    ax.view_init(elev=40, azim=frame_idx * 0.3 % 360)
    ax.set_zlim(0, 1.2)
    ax.set_axis_off()

    ax.text2D(0.5, 0.93, "◈ COMMENT BANGER ◈",
              transform=ax.transAxes,
              color='#ff88ff', fontsize=11, ha='center',
              fontfamily='monospace')

    fig.tight_layout(pad=0)
    img = _fig_to_array(fig)
    plt.close(fig)
    return img


STYLES = {
    "bars3d":  render_frame_bars3d,
    "galaxy":  render_frame_galaxy,
    "tunnel":  render_frame_tunnel,
    "terrain": render_frame_terrain,
}


# ──────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────

def _fig_to_array(fig) -> np.ndarray:
    """Convert matplotlib figure to uint8 RGB numpy array."""
    import matplotlib.pyplot as plt
    from PIL import Image
    import io

    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0,
                facecolor=fig.get_facecolor())
    buf.seek(0)
    img = Image.open(buf).convert('RGB')
    return np.array(img)


def frames_to_video(frames: list, audio_path: str, output_path: str, fps: int):
    """Write frames to MP4 and mux the original audio."""
    try:
        from moviepy.editor import ImageSequenceClip, AudioFileClip
    except ImportError:
        sys.exit("❌ pip install moviepy")

    print(f"\n🎬 Encoding {len(frames)} frames → {output_path}")
    clip  = ImageSequenceClip(frames, fps=fps)
    audio = AudioFileClip(audio_path)

    # Trim to shortest
    duration = min(clip.duration, audio.duration)
    clip  = clip.subclip(0, duration)
    audio = audio.subclip(0, duration)

    clip = clip.set_audio(audio)
    clip.write_videofile(output_path, fps=fps, codec='libx264',
                         audio_codec='aac', logger=None)
    print(f"✅ Video saved: {output_path}")


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="🎬 3D Audio Visualizer — renders your WAV as a 3D animated video",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
        Styles:
          bars3d   — classic 3D equalizer bars (rotating)
          galaxy   — spinning spiral galaxy that reacts to frequencies
          tunnel   — psychedelic wormhole tunnel
          terrain  — 3D audio mountain terrain (lava-coloured)

        Examples:
          python visualizer.py --input comment_banger.wav
          python visualizer.py --input comment_banger.wav --style galaxy --fps 30
          python visualizer.py --input comment_banger.wav --style terrain --res 1280x720
        """)
    )
    parser.add_argument("--input",  required=True,   help="Input WAV file")
    parser.add_argument("--output", default=None,    help="Output MP4 file (default: <input>_viz.mp4)")
    parser.add_argument("--style",  default="galaxy",
                        choices=list(STYLES.keys()),
                        help="Visual style (default: galaxy)")
    parser.add_argument("--fps",    default=24, type=int, help="Frames per second (default: 24)")
    parser.add_argument("--res",    default="1280x720", help="Resolution WxH (default: 1280x720)")
    parser.add_argument("--bins",   default=64,  type=int, help="FFT frequency bins (default: 64)")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        sys.exit(f"❌ File not found: {args.input}")

    width, height = map(int, args.res.lower().split('x'))
    output = args.output or args.input.replace('.wav', f'_{args.style}.mp4')

    print("\n" + "━"*50)
    print("   🎬 3D Audio Visualizer")
    print("━"*50)
    print(f"   Input  : {args.input}")
    print(f"   Style  : {args.style}")
    print(f"   Output : {output}")
    print(f"   FPS    : {args.fps}  |  Resolution: {width}×{height}")

    # Load audio
    print("\n🔊 Loading audio...")
    audio, rate = load_audio(args.input)
    duration = len(audio) / rate
    print(f"   Duration: {duration:.1f}s  |  Sample rate: {rate} Hz")

    # Analyse
    print("📊 Analysing frequencies...")
    fft_frames = get_fft_frames(audio, rate, args.fps, args.bins)
    rms_frames = get_rms_frames(audio, rate, args.fps)
    n_frames   = min(len(fft_frames), len(rms_frames))
    print(f"   {n_frames} frames to render")

    # Render
    render_fn = STYLES[args.style]
    frames = []

    try:
        from tqdm import tqdm
        iterator = tqdm(range(n_frames), desc=f"🎨 Rendering {args.style}")
    except ImportError:
        iterator = range(n_frames)

    print(f"\n🎨 Rendering frames (first frame may be slow due to import)...")
    t0 = time.time()
    for i in iterator:
        frame = render_fn(fft_frames[i], float(rms_frames[i]), i, width, height)
        frames.append(frame)

    elapsed = time.time() - t0
    print(f"✅ Rendered {n_frames} frames in {elapsed:.1f}s ({elapsed/n_frames:.2f}s/frame)")

    # Encode
    frames_to_video(frames, args.input, output, args.fps)
    print(f"\n🎉 Done! Open: {output}")


if __name__ == "__main__":
    main()