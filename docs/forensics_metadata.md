# Document Forensics — Metadata Analysis

Module: `src/forensics/metadata.py`

## What this module does

Reads the structural and EXIF metadata already embedded in a claim-document
image and turns it into structured metadata plus a small set of
presence/absence and consistency indicators. It does not compare against any
external reference and does not decide whether an image is genuine.

## How this implementation works

1. Load the image (path or in-memory `PIL.Image`) without ever writing to or
   mutating the source file. `img.format` is captured explicitly, since
   `PIL.Image.copy()` does not carry it over.
2. Read structural properties: format, width/height, color mode, and (for
   path input only) file size in bytes.
3. Read EXIF tags via `Image.getexif()` (top-level IFD0: `Make`, `Model`,
   `Software`, `DateTime`, ...) plus the Exif sub-IFD (`get_ifd(ExifTags.IFD.
   Exif)`, e.g. `DateTimeOriginal`). Each tag block is parsed independently
   inside its own `try`/`except`, so a corrupt sub-block does not prevent
   well-formed fields elsewhere from being returned — `exif_malformed` records
   whether any part failed to parse.
4. Derive boolean/numerical indicators from what was actually found (see
   below). No indicator alone is a fraud decision.

## Extracted structured metadata

- `format`, `width`, `height`, `color_mode`, `file_size_bytes` (`None` for an
  in-memory image with no known source file).
- `camera_make`, `camera_model`, `software`, `datetime_original`,
  `datetime_modified` (each `None` if absent).
- `exif_fields` — every EXIF tag found, as a flat `tag_name -> string` dict.

## Forensic features (`MetadataFeatures`)

- `width`, `height`, `aspect_ratio`, `file_size_bytes`.
- `has_exif`, `exif_field_count` — presence/volume of EXIF data.
- `has_camera_info` — `Make` or `Model` present.
- `has_software_tag` — `Software` tag present.
- `software_editing_indicator` — `Software` value contains a known
  general-purpose image-editor name (e.g. "Photoshop", "GIMP"). A simple
  case-insensitive substring check against a small fixed list — easily
  spoofed or absent even when editing did occur.
- `has_datetime_original`, `has_datetime_modified` — EXIF `DateTimeOriginal`
  / `DateTime` presence.
- `datetime_original_modified_mismatch` — both present and unequal.
- `exif_malformed` — any EXIF block failed to parse cleanly.

## Interpretation

These are descriptive indicators about what metadata is or is not present,
and whether it is internally consistent — not evidence ranked by severity.
Common, entirely legitimate reasons an indicator can fire:
- **Missing EXIF** is normal for screenshots, scans, PNGs in general, and
  images re-saved or recompressed by messaging apps/browsers, which commonly
  strip EXIF entirely.
- **Editing-software tag** can appear after legitimate cropping, rotation, or
  compression through a general-purpose tool — not only after fraudulent
  edits.
- **DateTime mismatches** can result from ordinary batch processing,
  cloud-storage re-encoding, or timezone handling — not only from tampering.

## Limitations

- **Metadata analysis is a forensic signal, not standalone proof of image
  manipulation.** It must be combined with other evidence (ELA, copy-move
  detection, structured claim signals, graph signals) in the downstream risk
  model — never used alone to label a claim as fraudulent.
- EXIF tags are simple, easily edited or stripped values; their absence does
  not imply tampering and their presence does not imply authenticity.
- No fraud/tamper threshold is hard-coded anywhere in this module. Any such
  decision belongs to a later, explicitly-scoped model.
- Only JPEG, PNG, TIFF, and BMP source formats are supported (validated
  against the image's detected format).
- Processes one image at a time; no batch API in this module.
