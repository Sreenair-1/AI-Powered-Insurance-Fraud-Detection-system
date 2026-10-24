// Client-side mirror of the API's upload rules (api/main.py), for early feedback only;
// the API re-checks every upload.
export const ACCEPTED_TYPES = ".png,.jpg,.jpeg,.tif,.tiff,.bmp,.pdf";
const ACCEPTED_SUFFIXES = ACCEPTED_TYPES.split(",");
export const MAX_UPLOAD_BYTES = 20 * 1024 * 1024;

export function validateUpload(file: File): string | null {
  const name = file.name.toLowerCase();
  if (!ACCEPTED_SUFFIXES.some((s) => name.endsWith(s))) return "Unsupported file type. Use PNG, JPEG, TIFF, BMP or PDF.";
  if (file.size > MAX_UPLOAD_BYTES) return "File is larger than the 20 MB limit.";
  if (file.size === 0) return "The file is empty.";
  return null;
}
