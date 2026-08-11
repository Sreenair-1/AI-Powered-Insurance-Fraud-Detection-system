# Document Forensics — Copy-Move Detection

Module: `src/forensics/copy_move.py`

## What copy-move detection is

Copy-move forgery detection looks for regions *within the same image* that
duplicate another region of that same image — e.g. a form field, stamp, or
signature copied and pasted to a second location to hide or alter content.
Unlike ELA (which compares against a recompressed copy), copy-move detection
compares the image against itself.

## How this implementation works

1. Load the image (path, in-memory `PIL.Image`, or an OpenCV-style numpy
   array — grayscale, BGR, or BGRA) without ever writing to or mutating the
   source.
2. Convert to grayscale and detect ORB keypoints + binary descriptors
   (`cv2.ORB_create(nfeatures=max_keypoints)`).
3. Self-match: match every descriptor against every descriptor in the same
   image with a brute-force Hamming matcher (`cv2.BFMatcher`), discarding
   the trivial self-match (a keypoint matching itself).
4. Apply Lowe's ratio test between the best and second-best remaining
   candidates per keypoint to discard ambiguous/weak matches.
5. Discard matches whose two keypoints are closer than `min_distance`
   pixels — nearby keypoints in ordinary local texture commonly have
   similar descriptors and are not evidence of copy-move.
6. Deduplicate symmetric pairs and summarize the remaining matches into
   numerical features, including the size of the largest cluster of matches
   that share an approximately equal spatial offset (many matches with the
   same offset indicate a whole block was translated, not just a few
   coincidentally similar keypoints).
7. Optionally render a visualization connecting matched keypoint pairs with
   lines on a copy of the image.

## Parameters

- `max_keypoints` (int, `> 0`, default `500`): Maximum ORB keypoints to
  detect.
- `ratio_threshold` (float, `(0, 1]`, default `0.75`): Lowe's-ratio-test
  threshold used to reject ambiguous matches; lower is stricter.
- `min_distance` (float, `> 0`, default `20.0`): Minimum pixel distance
  between two keypoints for their match to be kept.
- `offset_bin_size` (float, `> 0`, default `5.0`): Bin width in pixels used
  to cluster matches by spatial offset when computing `max_cluster_size`.
- `visualize` (bool, default `True`): Whether to render the match
  visualization image.

## Numerical features (`CopyMoveFeatures`)

- `num_keypoints` — ORB keypoints detected in the image.
- `num_raw_matches` — keypoints that had at least two candidate self-matches
  to run the ratio test on (before ratio/distance filtering).
- `num_filtered_matches` — matches remaining after the ratio test, minimum
  spatial distance filter, and symmetric-pair deduplication.
- `match_ratio` — `num_filtered_matches / num_keypoints`.
- `mean_match_distance`, `std_match_distance` — spatial distance (pixels)
  between matched keypoint pairs.
- `max_cluster_size` — size of the largest group of filtered matches that
  share an approximately equal spatial offset.

`CopyMoveResult.potential_duplicate_regions_found` is `True` when
`num_filtered_matches > 0`. This reflects only whether the matching and
filtering pipeline found candidate duplicated keypoint pairs — it is not a
fraud/tamper decision and carries no hard-coded fraud threshold.

## Interpretation

A higher `num_filtered_matches` or `max_cluster_size` is a signal worth
downstream attention — it suggests the image may contain a duplicated
region. It does not by itself indicate why the duplication exists: repeated
background patterns, printed form grids, and other legitimate repetitive
content can also produce matched keypoints.

## Limitations

- **This is a forensic signal, not standalone proof of image manipulation.**
  It must be combined with other evidence (ELA, metadata analysis,
  structured claim signals, graph signals) in the downstream risk model —
  never used alone to label a claim as fraudulent.
- ORB matching can miss copy-move forgeries that were rotated, heavily
  rescaled, or occurred in low-texture/flat regions with too few keypoints
  to match.
- Repetitive or patterned legitimate content (grids, textures) can produce
  matched keypoints without any forgery — `max_cluster_size` helps but does
  not eliminate this.
- Only JPEG, PNG, TIFF, and BMP source file formats are supported.
- Processes one image at a time; no batch API in this module.
