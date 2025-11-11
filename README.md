# video_dither
Small vibe coding project for video dithering with multiple methods

# Halftone Video Processor

A **memory-efficient Python tool** to apply multiple **dithering techniques** to video frames, creating stylized halftone-style videos with colored dot patterns.

[https://github.com/rggassner/video_dither/](https://github.com/rggassner/video_dither/)

---

## Features

* Supports multiple dithering algorithms:

  * `halftone`
  * `floyd-steinberg`
  * `atkinson`
  * `random`
  * `burkes`
  * `stucki`
  * `jjn` (Jarvis, Judice, and Ninke)
  * `bayer8x8`
  * `clustered`
* Choose output color: **red**, **green**, **blue**, or **white**.
* Resolution scaling for faster processing and smaller output.
* Preserves the original audio track.
* Minimal memory footprint (frame-by-frame processing).

---

## How It Works

Each frame of the input video is:

1. Scaled down to reduce resolution.
2. Converted to grayscale.
3. Dithered using the chosen algorithm.
4. Recolored to a single chosen tone.
5. Reassembled into a new video, with original audio reattached.

The output has a retro halftone / dot-art appearance — ideal for creative or lo-fi visual effects.

---

## Requirements

* Python 3.7+
* OpenCV (`cv2`)
* NumPy
* FFmpeg (must be available in PATH)

Install dependencies:

```bash
pip install opencv-python numpy
```

---

## Usage

```bash
python3 video_dither.py --input input.mp4 --dither floyd-steinberg --scale 4 --color blue
```

### Arguments

| Argument   | Type | Default    | Description                                            |
| ---------- | ---- | ---------- | ------------------------------------------------------ |
| `--input`  | str  | required   | Path to input video file                               |
| `--dither` | str  | `halftone` | Dithering algorithm (see list above)                   |
| `--scale`  | int  | `2`        | Resolution reduction factor (`2` → quarter resolution) |
| `--color`  | str  | `green`    | Output dot color: `red`, `green`, `blue`, `white`      |

Example:

```bash
python3 video_dither.py --input sample.mp4 --dither jjn --scale 8 --color red
```

This will produce:

```
output_jjn_red_8x.mp4
```

---

## Output Example

The script prints informative messages while processing:

```
[INFO] Extracting audio...
[INFO] Processing frames using floyd-steinberg dithering in color blue...
[INFO] Processed 100 frames...
[INFO] Muxing with audio...
[DONE] Saved output_floyd-steinberg_blue_4x.mp4
```

---

## Dither Methods Overview

| Method              | Description                                            |
| ------------------- | ------------------------------------------------------ |
| **halftone**        | Uses a Bayer matrix for classic print-style halftoning |
| **floyd-steinberg** | Error-diffusion with natural grayscale transitions     |
| **atkinson**        | Softer, denser diffusion pattern                       |
| **random**          | Random threshold noise pattern                         |
| **burkes**          | Efficient 3x5 error diffusion                          |
| **stucki**          | Smooth and detailed tonal transitions                  |
| **jjn**             | Fine-grained Jarvis–Judice–Ninke kernel                |
| **bayer8x8**        | High-quality ordered dithering                         |
| **clustered**       | Stylized clustered dot pattern                         |

---

## Cleanup

Temporary files (`temp_audio.aac`, intermediate video) are automatically removed after completion.

---

## Tips

* Try `--scale 8` for ultra-low-resolution retro results.
* Combine with color grading filters in FFmpeg for artistic looks.
* Use short clips first to test performance and visual style.

---

## License

MIT License © 2025 — Your Name

You are free to use, modify, and distribute this software as long as attribution is preserved.
