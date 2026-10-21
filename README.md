# Cutout Studio

Two tools in one app, both running **entirely in the browser**: background removal, and an overlay-removal / inpainting tool (switch with the *Background | Overlays* control in the top bar, or open `/?mode=wm`).

Background removal: Images are decoded, segmented by an AI model
(ONNX Runtime Web, WebGPU with a CPU/WebAssembly fallback), refined, edited, composited and exported on your own machine.
There is no backend and no image upload.

React 19 · TypeScript · Vite · Tailwind CSS 4 · onnxruntime-web · Zustand · fflate.

## Installation

Requirements: [Node.js](https://nodejs.org) 22.12 or newer (24 recommended), npm, and a current Chrome/Edge (WebGPU) — Firefox/Safari work on the slower CPU path.
No Python, no backend and no GPU drivers to install.

```bash
git clone https://github.com/rohitindustry09/Watermark-and-Bg-Remover.git
cd Watermark-and-Bg-Remover
npm ci
npm run dev
```

Open <http://127.0.0.1:5173>. `npm ci` installs the locked dependencies and copies the OCR files into `public/ocr` (a `postinstall` step), so run it before the first `npm run dev`.

- Background remover: <http://127.0.0.1:5173/>
- Watermark / overlay removal: <http://127.0.0.1:5173/?mode=wm> (or use the **Background | Overlays** switch in the top bar)

On Windows PowerShell the same commands work as written. If `npm run dev` complains about the Node version, check `node -v` and upgrade.

AI models are **not** in the repository: each is downloaded once by your browser on first use (MODNet 25 MB, BiRefNet-lite 109–214 MB, RMBG-1.4 42–168 MB,
LaMa 198 MB) and cached. Internet access is needed for that first download.

Production build and tests:

```bash
npm run build      # type-check + production build into dist/
npm run preview    # serve dist/ at http://127.0.0.1:4173
npm test           # unit tests
```

Docker: `docker compose up --build` serves the built app at <http://localhost:8080>.

## Overlay removal (watermark / label / timestamp inpainting)

For images **you own or are authorised to edit**: draft stamps, review labels, timestamps, your own watermarks. The app requires an authorisation
tick before it will restore or export, treats copyright/credit/stock-agency text as a protected category that needs a separate per-region
confirmation, and never selects anything automatically. It is not meant for removing someone else's ownership, attribution or licence marks.

**Workflow:** load images → OCR proposes text candidates (nothing selected) → tick the ones to remove, or draw a rectangle/lasso/brush →
inspect the red mask → *Restore selected* → compare with the split slider → export (original resolution, new file).

| Stage | What it really does |
|---|---|
| Detection | Tesseract.js OCR (Apache-2.0, self-hosted in `public/ocr`, English only), optional rotated scan (±15/30/45°). Each text line becomes a candidate with a *pixel* mask from local-contrast segmentation, not just its box. |
| Classification | Heuristics only: ownership/credit terms, timestamps, draft/sample terms, repeated faint text. Everything else is "original text / unknown". It never decides to remove anything. |
| Repeated marks | *Find similar marks* (normalised cross-correlation) and *Reference…* (your own overlay image, 5 scales). Matches are listed for review, unselected. |
| Manual | Add/Exclude brush, rectangle, lasso (with *Snap to overlay pixels*), click a region, undo/redo, reset. Exclusions always win. |
| Mask | Expand/contract, hole filling, overlay opacity. Mask is built at working resolution (1600/2560/4096 px by profile). |
| Inpainting | **Telea** fast-marching (own TypeScript implementation, runs in a worker) for small regions; **LaMa** (big-lama, Apache-2.0, 198 MB, 512 px windows, CPU/WASM) for larger ones; *Automatic* picks per region (≤ 40 px → Telea). |
| Blending | Inward feather (inside the mask only) and optional colour matching to the ring around each region. Both change the output; there are no decorative controls. |
| Export | Re-applied to the **original-resolution** file: only pixels under the (resampled) mask change. PNG/JPEG/WebP, filename suffix, never overwrites the source. Batch ZIP with `restoration-report.csv`. |

### What was measured (not marketing)

Test: a public-domain portrait with semi-transparent white text ("SAMPLE / PREVIEW ONLY", 55% opacity) added synthetically, so the true pixels are known.

| | Result |
|---|---|
| Mask recall of true overlay pixels | 100% (precision 52% because of the 3 px expansion) |
| Pixels changed outside the mask (working res) | 0 |
| Original-resolution export (2523×3313, 1950×2560 working) | 2.64% of pixels changed, all in the watermark area |
| PSNR on overlay pixels vs. truth | marked 9.6 dB → Telea 16.2 dB → LaMa 16.5 dB |
| Time | Telea 2.1 s; LaMa 89 s on CPU (first run also downloads 198 MB) |

PSNR is a weak measure for plausibility and low here because a face with a thick, opaque-ish overlay cannot be recovered; the filled area is an **estimate**.
Treat any reconstruction under an opaque overlay as invented content, never as recovered original pixels. Automated unit tests use synthetic images with known
ground truth (`tests/wm.test.ts`, `tests/wm-pipeline.test.ts`).

### Optional automatic watermark removal in the background tool

In the **Background** tool, the left panel has **Remove watermarks first** (**off by default**; it also needs the "I own these images or am authorised" tick).
When on, each *new* image is scanned before the cutout is made: OCR text lines classed as watermark / draft, and uniform logo-or-stamp marks in the corners and top/bottom
strips. The matches are reconstructed (Telea for small ones, LaMa for larger ones) and the cleaned PNG replaces the working file; the untouched upload is kept
and used again if you switch the option off and reprocess. Each image shows a scan report (candidates found, removed, left alone and why).

Two further switches appear once the option is on (both off by default): **Thorough** (remove every detected text line and mark, scan rotated text, use a more sensitive
logo detector, look for repeated copies of the marks it found, and include fragments next to a removed line) and **Include copyright / credit marks (I own them)**.
Thorough still refuses OCR noise: text that is not plausible, or not uniformly lighter/darker than its surroundings, is left alone. If nothing was removed from an image, its
panel says so and offers **Open in Overlays tool** for manual selection.

Measured with Thorough + credits on the synthetic test (draft stamp, 50% logo strip, © credit on a gradient with a coloured subject): 0% of the true overlay pixels remained,
PSNR on them rose from 8.4 to 48.1 dB, and 0 subject pixels changed (1.7% of other background pixels were re-filled at 48 dB). The first thorough version rewrote part of the subject
(OCR noise treated as text), which is why the consistency filter exists. Real photos are harder than this test.

What it will **not** do: it cannot find every watermark. Faint, patterned or full-image tiled marks, curved text and non-English text are often missed, and OCR only reads
clean, reasonably high-contrast text. It also never auto-removes copyright/credit/agency text (including when such text shows up inside a corner mark: the whole mark is left
alone and reported), timestamps, or ordinary scene text. Everything it fills is an estimate. For anything it misses, use the **Overlays** tool and select the region yourself.

Measured on a synthetic test (gradient photo, 85% white "DRAFT COPY", a 50% logo strip and a "© ACME Studio" credit): the draft stamp was removed, the credit was left
and reported; on the removed pixels PSNR against the true image rose from 10.4 dB to 24.8 dB (smooth background; textured areas will fare worse).

### Watermarks over the object, and background showing through it

Two problems were reproduced and fixed (see the numbers below):

1. **Background visible inside the subject.** The hole filler used to turn *any* enclosed background gap (between an arm and the body, inside a handle) into subject, whatever
   the model had said. It now fills only holes the model was unsure about; confident gaps stay removed. A new **Colour wand** tool (`W`) removes any background that still
   shows inside or over the subject: click it, adjust the tolerance, and optionally untick *Only connected pixels* to remove that colour everywhere.
2. **A big translucent watermark across the subject was not found, so cleanup happened only *around* it.** OCR cannot read large see-through text over a photo, and the corner-mark
   detector only looks at the edges. There is now an optional trained detector (**Find watermarks (AI)** in the Overlays tool, **AI watermark detector** in the Background tool)
   that returns boxes with a confidence; the overlay pixels are then extracted from inside each box (a multi-scale top-hat, light or dark detected automatically) and
   filled with LaMa. It is an optional 11 MB download (`ayan4m1/Watermark-Detection-YOLO11-ONNX`, **AGPL-3.0**, so check that licence suits your use), never used unless you switch it on.

**Recreate clean image (no watermark, no background)** in the Background tool turns on Thorough + the AI detector and re-runs the selected image: the watermark is removed first
(so it can no longer punch holes in the cutout), then the background is removed from the cleaned image.

Measured on a public-domain portrait with a 130 px rotated, 50%-opacity white "SAMPLE" stamped across the jacket (known ground truth):

| | Watermark option off | Recreate clean |
|---|---|---|
| Exported cutout vs. the true photo, mean error (0–255) | 11.2 | 3.8 |
| Pixels in the watermark area that are visibly wrong | 25% | 4% |
| OCR alone found the watermark | no (read fragments such as "LE", "VE") | — |
| AI detector found it | yes, confidence 0.82; no false detection on the clean photo | — |
| Overlay-pixel PSNR (marked → restored) | 9.3 dB | 19.6 dB |

Honest limits: thorough mode with LaMa is slow on the CPU (tens of seconds to minutes per image); the fill on a textured, detailed photo is an estimate and shows softness;
the detector can miss stylised marks or flag non-marks, so always review the result; the fast MODNet portrait model fragments some subjects (use RMBG-1.4 for personal use or
BiRefNet-lite on a working WebGPU device for better cutouts).

#### Coloured (tinted) watermarks on product photos

Real-world case: a faint orange/red "Dharma Statuos" watermark across the base of a brass statue. OCR read only garbage from the statue's relief, the trained detector found nothing
(it was not a brightness overlay), and the corner-mark detector mistook the statue's own bright finials, mace and foot for logos and would have inpainted them. So:

- A new **colour detector** (no model download, on by default for watermark removal) handles images dominated by one colour family (brass, silver, wood, a single-colour garment).
  It looks for rare saturated pixels whose hue differs from the image's dominant hue and that cluster into a text-like band, then grows the match along the same text line.
  It stays silent on grey images, colourful images (natural colour is protected), clean images and isolated specks.
- The corner/edge-mark detector is now **Thorough-only**, with a size limit (marks must be compact). The default pass uses OCR + the colour detector (+ the AI detector if switched on).

Measured on that statue photo with default settings: 1 region found and removed (≈43 s, LaMa); only 0.92% of pixels changed, all inside the watermark's own box, and the rest of the image
is untouched. Before the corner-detector fix the same run rewrote 3.6–7.7% of the image, including parts of the statue.
The fill is an estimate: the reconstructed ropework on the base is darker and softer than the real surface.

### Image enhancer: 4K upscaling and quality boost (both tools)

The **Enhance & upscale (4K)** panel (right side, above Export; **off by default**) improves the final, recreated image **when you export it**, so what you get is the clean image
(watermark removed, background applied) at the chosen size. It never changes the picture's geometry.

| Setting | What it does |
|---|---|
| Resolution | Keep size · 2K (long edge 2560) · 4K (long edge 3840). Images already large enough are not upscaled; a note says so. |
| Upscaling method | **Classic** (smooth resampling, seconds) · **Auto** (AI when it needs ≤ 260 tiles ≈ 4–5 min, otherwise classic, with a note) · **AI** (always). |
| AI strength | Mix of the AI result and the natural classic upscale (default 70%). Lower looks more natural; higher is sharper but can look "painted" on metal and specular highlights. |
| Denoise | Edge-preserving noise reduction, applied *before* upscaling so noise is not enlarged. |
| Sharpen | Unsharp mask at the output size (radius scales with the upscale factor). |
| Auto tone | Gentle automatic levels, capped so dark or bright products are not blown out. |

AI upscaling uses **Real-ESRGAN general-x4v3** (BSD-3-Clause, ≈ 5 MB, downloaded once from Hugging Face, runs in the browser on the CPU). The network takes fixed 128×128 tiles; the image is
processed tile by tile and only each tile's context-free core is kept, written directly into the output canvas, so a 4K result never needs a huge intermediate image. Transparent images keep a
verified alpha channel: colours under transparent pixels are bled from the subject first (no halos), alpha is resampled separately, and the exported PNG is re-checked as before.

Measured on a 874×763 brass-statue photo → 3840×3352:

| | Classic | AI |
|---|---|---|
| Time | 6.9 s | 89 s (72 tiles, ≈ 1.2 s/tile) |
| Detail (Laplacian variance on the faces) | 17.8 | 255.6 |

Full app path (process → 4K enhance with AI → export PNG): 10.4 MB RGBA file, corner alpha 0, centre alpha 255, alpha verified, 70 s.

Honest limits: AI upscaling **invents** plausible detail, it does not recover lost pixels, and it can over-smooth or "paint" fine texture (use the AI strength slider). WebGPU was no faster than the CPU for this
network on the test machine, so it runs on the CPU; big sources mean many tiles (a 2400×2400 image is ~530 tiles, roughly 10 minutes), which is why Auto falls back to classic there.
Large 4K PNGs are tens of MB; use WebP/JPEG for smaller files.

### Limits you should know about

- **OCR is weak on faint overlays.** On the 42–55% opacity test it misread "SAMPLE" and found nothing usable; it reads clean, high-contrast text well
  (96% / 95% on a clean test image). Faint, decorative, curved or non-English text needs the manual tools; *Snap to overlay pixels* then isolates the overlay inside your shape.
- Logo detection is template matching, not a trained detector; it works for near-identical repeats, not for arbitrary logos.
- No face/object model: the app warns about large masks and high-detail surroundings but cannot know that a face is underneath.
- LaMa runs on the CPU (the WebGPU provider handles its FFT ops poorly; try `?lama=gpu` at your own risk). It is slow and needs ~200 MB of download and a lot of memory.
- Only English OCR data is bundled. TIFF export is impossible in browsers; HEIC/TIFF import only works where the browser can decode it.
- No server-side job queue or CUDA path: everything is client-side by design.

## Quick start (Windows, macOS, Linux)

Requires Node.js 22.12+ (24 recommended) and a current Chromium-based browser (Chrome/Edge 113+ for WebGPU; Firefox/Safari
work on the CPU path).

```bash
cd bg-studio
npm ci          # also copies the OCR assets into public/ocr
npm run dev
```

Open <http://127.0.0.1:5173>. The first time you process an image the browser downloads the model (25–214 MB, one time,
cached in Cache Storage). Production build:

```bash
npm run build
npm run preview      # http://127.0.0.1:4173
```

Tests and type-check:

```bash
npm test
npm run typecheck
```

## Sample workflow

1. Drop a JPG/PNG/WebP (or paste with Ctrl+V, or use Add images / Folder / ZIP / URL).
2. Pick a profile: **Speed** (MODNet), **Balanced** or **Max quality** (BiRefNet-lite where WebGPU works).
3. Wait for the status chip on the thumbnail to reach *Done*. The canvas shows the cutout on a checkerboard.
4. Compare with **Split** (`\`) or **Original** (`C`). Turn on *Show uncertain edges* to see where the model was unsure.
5. Fix mistakes: **Keep brush** (`B`), **Erase brush** (`E`), **Object** (`O`) to remove or keep a whole region, **Rectangle** (`R`),
   **Lasso** (`L`). `Ctrl+Z` / `Ctrl+Shift+Z` undo and redo. *AI mask* returns to the untouched model result.
6. Tune **Edge refinement**: fine-detail recovery, expand/contract, feather, smoothing, colour decontamination.
7. Choose a background (none, colour, gradient, blurred original, image), shadow, canvas size and placement, or enable **Product photo mode**.
8. **Export** as PNG / WebP / JPEG. The app decodes the exported file again and verifies that the alpha channel survived.
   For many images use **All as ZIP** (optional `report.csv`).

## Models

| Model | Licence | Size | Notes |
|---|---|---|---|
| BiRefNet-lite (`onnx-community/BiRefNet_lite-ONNX`) | MIT | 109 MB fp16 / 214 MB fp32 | Best detail. Needs WebGPU: on the WebAssembly backend the 1024×1024 graph runs out of memory, so MODNet is used instead. |
| MODNet (`Xenova/modnet`) | Apache-2.0 | 25 MB | Fast portrait matting. Weak on products, animals and unusual photos. |
| RMBG-1.4 (`briaai/RMBG-1.4`) | **Non-commercial** (BRIA licence) | 42–168 MB | Optional. Good general quality; never the default; the UI shows a warning. |

Weights are **not bundled**. They are fetched by the visitor's browser from `https://huggingface.co` on first use and cached.
To use a mirror, copy `.env.example` to `.env` and set `VITE_MODEL_HOST` (the mirror must send CORS headers and keep the
`<repo>/resolve/main/<file>` layout). To add a model, add an entry to `src/lib/models.ts` (input size, normalisation, output type).

Device selection: WebGPU when an adapter exists (fp16 when supported), otherwise multi-threaded WebAssembly. If a GPU run
fails or exceeds 60 s the app switches to CPU automatically. Overrides for debugging: `?device=cpu`, `?fp16=0`.

## How the pipeline works

1. **Decode** – `createImageBitmap` applies EXIF orientation; magic bytes are sniffed; limits: 60 MB, 80 MP. The original file is never modified.
2. **Segment** – the image is resized to the model input only for inference; the probability map is resampled back to the working resolution.
3. **Refine** (Web Worker) – guided filter against the full-resolution luminance (snaps the matte to real edges: hair, fur, lace),
   alpha curve (strength/contrast), enclosed-hole fill and speck removal, manual edits, expand/contract, smoothing, feather.
4. **Cutout** – straight-alpha RGBA. Soft-edge pixels get their colour re-estimated from nearby solid foreground to remove
   background colour halos.
5. **Composite** – background, optional drop shadow, scale/offset/canvas fit, rendered with Canvas 2D.
6. **Export** – re-rendered at the original resolution (alpha is upsampled and the original is re-decoded when it was downscaled
   for processing), encoded, then decoded again to verify transparency.

Working resolution per profile: 2048 px (Speed), 3072 px (Balanced), 4096 px (Max quality) on the long edge. Batches segment one
image at a time and keep only an 8-bit matte per image (oldest evicted beyond 24 images and recomputed on demand).

## Keyboard shortcuts

`W` colour wand · `V` pan · `B` keep brush · `E` erase brush · `O` object · `R` rectangle · `L` lasso · `[` `]` brush size · `Space` hold to pan ·
`0` fit · `1` 100 % · `+` `-` zoom · `\` split compare · `C` show original · `←` `→` previous/next image · `Ctrl+Z` / `Ctrl+Shift+Z` undo/redo.

## Deployment

It is a static site. Any static host works, but it **must send** these headers for multi-threaded WebAssembly:

```
Cross-Origin-Opener-Policy: same-origin
Cross-Origin-Embedder-Policy: credentialless
```

Without them everything still works, only the CPU path is single-threaded.

### Docker

```bash
docker compose up --build        # http://localhost:8080
```

`nginx.conf` sets the isolation headers, a Content-Security-Policy, and long-term caching for `/assets`. Adjust `connect-src` if you
mirror the models. (`img` URL import needs `https:` in `connect-src`; remove it if you disable that feature.)

### Linux VPS (nginx)

```bash
npm ci && npm run build
sudo mkdir -p /var/www/cutout && sudo cp -r dist/* /var/www/cutout/
sudo cp nginx.conf /etc/nginx/conf.d/cutout.conf    # change root to /var/www/cutout
sudo nginx -t && sudo systemctl reload nginx
```

Serve over HTTPS (browsers expose WebGPU, Cache Storage and `crossOriginIsolated` only in secure contexts; `localhost` counts as secure).

## Privacy and security

- Images never leave the browser. The only network traffic is the one-time model downloads (BiRefNet/MODNet/RMBG/LaMa from Hugging Face) and URL imports you trigger. OCR files are served from this site.
- Uploads are validated by magic bytes and by actually decoding them; file size, pixel count, ZIP entry count and expanded ZIP size are limited.
- ZIP entries are reduced to a safe basename (`../`, absolute paths and drive letters are dropped). CSV reports neutralise spreadsheet formulas.
- Nothing is persisted except the cached model weights. *Clear site data* in the browser removes them. No analytics, no logging of image content.

## Known limits (be aware before relying on it)

- Object labels: regions are separated from the matte only; the app does **not** classify objects (no "person/product/animal" labels),
  has no text-prompt selection, no SAM-style promptable segmentation, no human-parsing or depth models.
- TIFF export is impossible in browsers. HEIC/HEIF and TIFF import only work where the browser can decode them.
  Camera capture uses the device camera on phones/tablets (a file picker on desktop). Colour profiles are not read or preserved
  (output is sRGB as decoded by the browser).
- Glass/transparent objects, white-on-white and very low-contrast edges are limited by the models; use the uncertainty overlay and the brushes.
- BiRefNet-lite is GPU-only (see above) and hung on at least one AMD RDNA2 GPU during development; the CPU fallback then uses MODNet.
- Expand/contract uses an octagonal morphological kernel, not a true Euclidean distance.
- Projects are not saved between sessions.
- The Docker image was written but could not be built in the development environment (no running Docker daemon).
