# Document Forensics — Error Level Analysis (ELA)

Module: `src/forensics/ela.py`

## What ELA is

Error Level Analysis re-saves an image at a controlled JPEG quality and
measures the pixel-level difference between the original and the
recompressed copy. Regions of an image that were edited or pasted in after
its last "real" compression tend to show a different recompression error
level than regions that were untouched, because they carry a different
compression history. ELA surfaces that difference.

## How this implementation works

1. Load the image (path or in-memory `PIL.Image`) without ever writing to
   or mutating the source file.
2. Convert to RGB (grayscale, RGBA, and other source modes are normalized).
3. Re-save that RGB image as JPEG at a controlled `quality`, entirely in
   memory (`io.BytesIO`), then reload it.
4. Compute the absolute per-pixel difference between the original and
   recompressed arrays, using a signed `int16` intermediate representation
   to avoid `uint8` subtraction overflow.
5. Amplify the difference by `scale` (clipped to `[0, 255]`) to produce a
   human-viewable `ela_image`. The numerical features below are computed
   from the *unscaled* difference, so `scale` only affects visualization.
6. Summarize the difference into a small set of numerical features.

## Why JPEG recompression is used

JPEG's block-based, quantized DCT compression is what produces the
recompression-error signal ELA relies on. PNG/TIFF/BMP inputs are still
recompressed as JPEG purely to derive this signal — the result describes
how the image content responds to JPEG requantization, not the forensic
history of the source format's own (typically lossless) compression. This
is documented, not hidden: `metadata["original_format"]` in the result
records the true source format.

## Parameters

- `quality` (int, `1`–`100`, default `90`): JPEG recompression quality.
  Lower values increase the raw error magnitude across the whole image,
  which can help visualize subtle differences but makes numeric features
  less directly comparable across different quality settings.
- `scale` (float, `> 0`, default `10.0`): Visualization amplification
  factor applied to the difference before clipping to `[0, 255]`. Does not
  affect the numerical features.

## Numerical features (`ELAFeatures`)

Computed from the per-pixel mean absolute error across RGB channels
(`diff.mean(axis=-1)`), all on the unscaled `[0, 255]` error range:

- `mean_error` — average error across all pixels.
- `std_error` — standard deviation of error.
- `median_error` — median error.
- `max_error` — maximum single-pixel error.
- `p95_error` — 95th percentile of the error distribution.
- `high_error_pixel_ratio` — fraction of pixels with error above a fixed
  absolute threshold (30.0, on the unscaled 0-255 range). This is a
  descriptive statistic about the image's own error distribution, not a
  fraud/tamper decision rule.

## Interpretation

Higher or more spatially inconsistent error values are a signal worth
downstream attention — they suggest the image may contain regions with
different compression histories. They do not, by themselves, indicate why
that inconsistency exists: scans, screenshots, prior legitimate JPEG
re-saves, and low-quality source images can all elevate ELA error with no
forgery involved.

## Limitations

- **ELA is a forensic signal, not standalone proof of image manipulation.**
  It must be combined with other evidence (copy-move detection, metadata
  analysis, structured claim signals, graph signals) in the downstream risk
  model — never used alone to label a claim as fraudulent.
- No fraud/tamper threshold is hard-coded anywhere in this module. Any
  such decision belongs to a later, explicitly-scoped model.
- Only JPEG, PNG, TIFF, and BMP source formats are supported (validated
  against the image's detected format).
- Processes one image at a time; no batch API in this module.
