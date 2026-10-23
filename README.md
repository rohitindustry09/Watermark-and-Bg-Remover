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
| Inpainting | **Telea** fast-marching (own TypeScript implementation, runs in a worker) for small regions; **LaMa** (big-lama, Apache-2.0, 198 MB, 512 px windows, CPU/WASM) for larger ones; **Patch** (exemplar fill, below) copies real texture; *Automatic* picks per region (textured surroundings → Patch, otherwise ≤ 40 px → Telea, larger → LaMa). |
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
- The corner/edge-mark detector is **no longer part of the automatic pass** at all (it cannot tell a logo from a bright part of the subject), and neither is the automatic
  "find repeats" search (it matches real surface detail such as ropework and repaints it). Both remain available as manual, reviewed actions.
- Rotated OCR (Thorough) reads photo texture as short "words" ("ol", "32"). A rotated hit now has to read as real text (≥ 3 letters/digits) to be removed, and unclassified text needs
  OCR confidence ≥ 65 in Thorough mode.

Measured on that statue photo (990×990) with Thorough + the AI detector: before these fixes 19 regions were repainted, including 7+ patches on the statue's lap, legs and foot (4.4% of the
image); now 4 regions (the top-left logo and spec text, and the base watermark) and 2.0% of pixels, with the statue itself untouched.

#### Keeping detail sharp: texture-preserving fill

A smooth fill (Telea, or LaMa on repetitive micro-detail such as rope, petals or engraving) looks blurred next to real texture. The **Patch** method (`src/wm/patch.ts`, exemplar
inpainting after Criminisi et al. 2004) copies real patches from the surrounding image, so the fill keeps real sharpness. Plain exemplar copying alone produced a patchwork (texture
taken from unrelated places, a broken rim line), so it is **guided**: LaMa (or Telea if LaMa is unavailable) reconstructs the structure first; patch search must agree with it; and only the
*high-frequency detail* of the copied patches is added on top of the guide's colour and shading. Where the best patch matches poorly, the guide's own detail is used instead, so a bad copy
degrades to a smooth fill rather than to visible blocks. *Automatic* uses it when the pixels around a region have strong local contrast (≤ 40 000 masked pixels); you can also choose it explicitly.

On synthetic rope texture with a known answer the plain fill reproduced the hidden area with a mean error of 1.5/255 and the same contrast, where Telea lost about two thirds of the contrast.
On the statue's base the result keeps the rope detail and the rim line instead of a smooth smear. It is still an estimate: it cannot recover detail that was hidden under an opaque watermark, and the exact
look of the fill varies with the surrounding texture. The full Thorough run on the statue took 78–81 s on the CPU with the guided fill (59 s with the unguided version; 65–86 s before this change
when it repainted 5–19 regions; single runs each, not averaged).

### Product photos with brand watermarks (bowl and statue test)

Real case that exposed several problems at once: two catalogue photos with a translucent "DHARMA STATUES" overlay across a brass bowl, an orange "Statues" stamp on a silver statue's base,
and a logo block with spec text on the backdrop. Everything below was checked on those two images (990×990, CPU path, single runs):

- **Cutout holes and lost parts.** When BiRefNet-lite could not run (it needs WebGPU, and on a GPU that hangs it times out), the app fell back to MODNet, a *portrait* model. On the bowl it cut holes
  through the shiny interior and left a stray blob; on the statue it dropped both spires on the base. The fallback is now RMBG-1.4 (object-capable, 42 MB int8 build that runs on the CPU in the
  inference worker, about 8 s per image), which kept the whole bowl and both spires. Its licence is non-commercial, which the on-screen notice says; MODNet remains selectable for portraits.
- **A broken GPU is remembered.** A hung GPU used to cost a full timeout in every session. The failure is stored (14 days) so later sessions go straight to the CPU model; open the page with
  `?device=gpu` to try the GPU again. The GPU wait is also 30 s instead of 60 s. (BiRefNet-lite on WebGPU hung on the machine used for these tests as well.)
- **Stray bits.** Disconnected foreground pieces under 2% of the main subject are dropped (was 0.5%), except pieces within 1.5% of the image size of the subject, so thin tips and detached
  details survive.
- **Text the OCR cannot read.** Tesseract returned gibberish for the stylised overlay and the AI watermark detector found nothing. A text *detector* (PP-OCRv4 DBNet, Apache-2.0, 4.7 MB, runs in
  the inference worker) finds text-shaped marks without reading them: on the bowl it boxed the big translucent "DHARMA STATUES", the red tagline under it, the brand text and the spec lines. Its
  boxes are refined to the marks' own pixels with two extractors combined (one finds light or dark strokes, one finds local contrast), because a single extractor picked the gold highlights
  instead of dark red letters. It is selected only with confidence ≥ 90%, strokes that could be extracted, and a width of at least 8% of the image, so a few engraved letters on the product are
  not touched (Thorough lifts the size rule).
- **Coloured stamps on a similar-coloured scene.** A saturation detector finds marks that are far more saturated than the rest of the picture (the orange "Statues" stamp on a peach/silver
  scene) inside the border band, as small sparse shapes, and only if they stand clearly apart from their immediate surroundings. It stays silent on brass or a red wall.
- **The hue detector was fooled by a reddish wall and the figure's warm highlights** (six false regions, one repainted the figure's head). A hue mark now has to be a wide, short text line and be
  confirmed by the text detector, or be an unmistakable line with high confidence.
- **Subject first.** The subject is cut out first; only marks that touch it are repainted. Marks on the backdrop leave with it (7 of 10 on the bowl), which saves a fill each and cannot damage the
  subject. If you keep a blurred copy of the original background, every mark counts instead. The first cutout is reused after the repaint (re-cutting cost about a third of the time).
- **LaMa and memory.** LaMa is a 198 MB model that needs well over 1 GB once loaded. On a 5.5 GB PC with about 0.5 GB free, loading it killed the whole browser. Devices that report 4 GB or less now skip
  it in automatic mode (a switch in the Watermark panel, or `?lama=1` / `?lama=0`, overrides). Text strokes never need it anyway: thin strokes are filled with Telea and real texture patches.

- **Exact letters, not slabs (second pass on the bowl).** The first version left red and green specks and, at zoom, a mosaic over the carved ornament. Cause: the stroke masks covered 35-45% of
  each text box (relief around the letters counted as text), and growing them by 3-5 px merged everything into two slabs, which were then filled with texture tiles. Now (`src/wm/strokes.ts`):
  only features about one stroke wide count (thin top-hats), plus colour cues that survive shadow (a local hue difference and ln(R/G), because dark red letters on a dark red-brown carving barely
  differ in hue *angle*); candidates are ranked by strength and capped at a text-like share of the box (22%, also after closing gaps); each text region grows by about one stroke width on its own
  instead of the global 3-5 px; a hue-based mark that the text detector already covers is dropped (its blobs spilled over the carving); and a cluster that is *mostly* thin strokes is repaired with
  Telea only (texture patches tile visibly over blurred or glossy relief). Result at 4x zoom: no letters, no ghosts, no red or green remnants, ornament structure intact.

- **Faint haze on marbled backdrops.** A bronze statue photographed on a cloudy white backdrop kept 11% of the image as semi-transparent wisps (about 13% opacity on average, in 436 separate
  pieces), because the model is unsure about the texture and the opacity curve only clears values under about 14%. `suppressHaze` (`src/lib/matte.ts`, default on, a "Clear faint haze" slider)
  clears weak opacity away from the subject, and any piece that never becomes solid; everything within 1.5% of the image size of the confident subject is untouched, so soft edges and fine detail stay.
  On that file: 112,177 semi-transparent pixels became 3,198 (the subject's own soft edge), nothing near the subject changed, and it takes about 0.1 s.
- **Carving is not text.** On that statue the text detector scored two bands of the carved lotus-petal border at 94% and 91% and the earlier rules would have repainted them. Repeating carving is
  monochrome while real overlays scored 98-99% or are clearly coloured (35-45% of their pixels), so the default pass now needs a score of at least 97%, or a coloured mask with at least 85%
  (Thorough still takes 60%). A hue-based mark is only dropped in favour of a text box when that box is itself used.

#### Third pass: a bronze statue on a grey marbled backdrop (original photo available)

Run on the original, the earlier build still showed blotches of backdrop, a pale patch in the gap between the arm and the torso, a smeared crown and base, and the watermark on the base. Causes found by
measuring, and what changed:

- **Blotches.** The model's raw output was noisy on the marbled backdrop (about 47,000 pixels above 50% on the left half). The "near the subject" zone that protects soft edges was built from every solid
  pixel, so each blotch's solid core sheltered its own halo. It is now anchored to real objects only (pieces of at least 3% of the largest), the weak bridges tying a blotch to the subject are cleared first,
  and speck removal runs before the haze step. Faint pixels on the left: 70,574 without the rule, 13,421 with the first version, 0 now.
- **Holes and gaps.** Hole filling treated any enclosed gap the model was unsure about as part of the object, which is right for a glossy bowl interior but wrong for the gap between an arm and a torso. A
  gap whose colour matches the backdrop around the subject now stays transparent, and a large detached piece that looks like the backdrop is removed. The backdrop colour comes only from background connected
  to the image edge (an enclosed gap has zero opacity too and must not define "the backdrop").
- **Smeared crown and base.** The saturation detector had an absolute floor of 45% and the statue's own warm highlights sit around 50-55%, so they were reported as "logos" and repainted. The floor is now
  62%, a mark on the subject is compared only with the rest of the subject, and its mask gets a thin band for the letters' anti-aliased edges. With that, the bronze statue's real watermark along the base
  (red-orange "Dharma Statues") is found, which it was not before.
- **Repair borrowing backdrop colour.** When the subject is known, the Telea repair treats the backdrop as unknown, so a mark at the subject's edge is not filled with pale backdrop.

Remaining on that photo: the repair along the base's bottom edge is soft and slightly smudged, a strip of grey backdrop under the base is kept by the cutout itself, and on the silver statue a faint orange
smudge of the stamp can remain. Two cues I tried to tell a real stamp on a metal subject from the subject's own highlights (shape thinness, contrast) did not separate them; only the saturation floor did.

Not solved: a mark that runs along the very edge of the subject (red "Dharma" and orange "Statues" along the bottom edge of that statue's base) is found only as a low-confidence colour candidate and
is left alone, because the repair there smeared the base in my test (checked on the cutout only; the original photo was not available). That case needs the original to tune.

Result on the two photos: the statue loses the "Statues" stamp, the spec lines and the badge and keeps both spires; the bowl loses both lines of the overlay, keeps its whole interior and the figure on
top, and has no stray blob. An 8-image mixed batch (both photos alternating, CPU path, roughly 0.5 GB of RAM free) finished with no crash in 90 s: 23 s for the first image and 8–11 s for each of the rest (it was about 20 s each before the cutout was reused).

Limits: the text detector can miss small or very faint marks (it missed the faint brand text beside the statue's badge, and the orange stamp, which the saturation detector catches instead), the fill under a mark that crosses fine relief
(the bowl's ornament) is an estimate, and RMBG-1.4's licence is non-commercial.

### Responsiveness and hangs

Measured on a 990×990 photo with watermark removal on (single runs on this CPU, so treat the ratios as indicative):

- **Page freezes.** CPU AI models (LaMa, the watermark detector, the CPU segmentation models) used to run on the page's main thread, which blocks it for the whole run: one LaMa fill froze the
  page for 23 s, and 44 s of a 65 s batch run was spent frozen. They now run in a dedicated inference worker (`src/lib/infer.worker.ts`), and the mask morphology that remained on
  the main thread was made O(pixels) instead of O(pixels × radius) with identical output (`src/lib/morphology.ts`, checked against the old code on random masks). The longest freeze fell
  from 23 s to under 1 s and the frozen share of a run from about two thirds to about 5%.
- **Faster as a side effect.** `clusterMask` 2294 → 79 ms, `ringDetail` 451 → 38 ms, `buildMask` 466 → 89 ms, `detectTint` 839 → 318 ms, `segmentVeil` 1568 → 414 ms. One LaMa fill took about 16 s
  in the worker against about 25 s on the main thread. The thorough run on the statue went from 29–37 s to 23–27 s warm. These costs scale with image area, so larger images gain more.
- **No more silent hangs.** A run that overruns is stopped for real: the inference worker is terminated (impossible on the main thread, where a timer cannot even fire while it is blocked) and the
  next image starts a fresh one (LaMa 240 s, CPU segmentation 300 s, detector 120 s). A model download with no data for 45 s, or no response in 60 s, fails with a clear message; an OCR pass
  over 90 s restarts the OCR worker; GPU model preparation over 120 s is abandoned; and the whole watermark stage of one image is capped at 8 minutes, after which that image continues without
  watermark removal and the batch moves on.
- **Fewer UI re-renders.** Download, tile and OCR progress updates are rate-limited at the source (a store update re-renders the UI), and the image strip no longer re-renders every tile on each change.

Not covered: the WebGPU path (BiRefNet) was not exercised here, so its behaviour on your GPU is unchanged apart from the new timeout on preparing the model; the stall and hang safety nets are
unit-tested with fake timers and workers, not triggered by a real network failure; the Overlays tool shares these modules but was not run live in this pass.

### Bulk batches: folders, shared marks, fast fill, resume

- **Folder structure in, same structure out.** A ZIP can contain folders nested to any depth: every image is found, and the export ZIP recreates the same folders
  (`shoot/day1/deep/hat.png` → `shoot/day1/deep/hat_cutout.png`), so same-named files in different folders stay separate. This also works for the **Folder** button
  (the picked folder's subfolders) and for folders dragged from your file manager. Hidden and system entries (`.git`, `.DS_Store`, `__MACOSX`), absolute or `..` paths and
  non-image files are skipped (non-images are counted in a note). The CSV report lists the full paths. A ZIP may hold up to 2000 images and 500 MB once unpacked; a ZIP inside a ZIP is not opened.
  Checked end to end in the app on a 5-image ZIP nested six folders deep; folder drag-and-drop is covered by code, not by a test I could run here.

For a ZIP or a multi-file drop with **Remove watermarks first** on (`src/wm/batch.ts`, `src/lib/resume.ts`):

- **Shared marks.** The first 3 images of each size group are fully analysed. A mark that appears in the same place in most of them becomes a *shared mark*. Later images of that
  group skip OCR and the AI detector: each shared mark is re-checked on the image's own pixels (the overlay is re-segmented in that box and must overlap the learned mask), and an image that
  fails the check falls back to full detection. After two failed checks a group stops sharing, so a batch with varying watermarks never skips detection wrongly. The colour (tint) detector is
  cheap and runs on every image. A clean image in a sharing group fails the check and is analysed in full, so it is correct but not faster.
- **Fast fill.** In *Automatic* mode a region whose surroundings are a plain backdrop (very low local contrast) is filled with Telea instead of LaMa; textured surroundings use Patch, others LaMa.
- **Resume.** Finished watermark-removal results and finished exports are stored in this browser (IndexedDB), keyed by the file's content fingerprint plus every setting that changes the
  output (a mask edit or background change re-renders correctly). If the tab crashes or closes mid-batch, add the same files again and finished steps are reused. Up to 400 entries
  (oldest evicted); a *Clear* link in the Watermark panel empties it. The cutout (matte) step itself is not saved: it is recomputed, which takes seconds per image.

- **Time left.** The footer shows `about 1 h 12 min left · done around 6:45 PM` while images process, and the same for the ZIP export. It uses the median of the recent finished
  images (the first, which carries model loading, is dropped once there are enough) and is marked *(rough)* until 6 images have finished. It is an estimate: a batch of mixed image
  sizes or a different watermark mid-way will move it. Checked live on 22 small images (it counted down correctly); the export estimate is unit-tested but I did not watch it live.

Measured on the 990×990 statue photo (single runs on the CPU, with the hidden-pane throttling that makes timings noisy): Thorough detection alone costs about 19 s of rotated OCR plus
4 s for the AI detector per image, which shared marks skip; the same run on the learned images reused the same 4 regions (mask 3.39% vs 3.30% on a noisy copy). End-to-end image times
varied between 37 s and 70 s because the LaMa/Patch fill dominates, so I do not claim a precise end-to-end speed-up. Parallel workers are not implemented.

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

#### Memory use and "out of memory" errors

Browsers cap a tab's memory (about 2.8 GB of JavaScript heap on the test machine). The enhancer therefore runs denoise, sharpen and edge-colour bleed in **bands** (about 1.5 megapixels each,
with overlap so the result matches a whole-image pass to within 1 level), which keeps peak memory independent of image size: a 24-megapixel image takes 26 s and an **81-megapixel image completes in 112 s**.
If memory still runs out, the export is not lost: the enhancer retries with only the cheap steps (no denoise, auto tone or AI), and as a last resort exports the image unenhanced with an explanation.
Out-of-memory failures anywhere else (segmentation, LaMa, full-size decode) are reported as a plain message with what to try: the Speed/Fast profile, a smaller image or export size,
turning off Thorough/Enhance, or closing other tabs. I did not capture an exact peak-heap figure (the page was too busy to sample it), so treat these numbers as "completed without failing" rather than a measured ceiling.

#### Responsiveness and blank-image protection

Enhancement runs in a **Web Worker**, so the page no longer freezes while it works (measured on a 2000×2000 → 3840×3840 classic enhance: the longest main-thread freeze fell from **7.9 s to 0.17 s**;
a production-build export through the real Download button with AI upscaling froze the page for at most 0.11 s). Progress and a **Cancel** button show in the panel.
Intermediate canvases are CPU-backed, which avoids the GPU read-back stalls and the silent loss of canvas contents that can happen under memory pressure.
Before an enhanced image replaces the export it is compared with the input (alpha and detail); if the result came back **blank, fully transparent or flattened to one colour**, it is discarded,
the unenhanced image is exported instead, and the panel says why. The background tool also refuses to save a composite that is fully transparent while a subject was detected.
If the worker cannot start, enhancement falls back to the main thread (slower, still correct).

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
