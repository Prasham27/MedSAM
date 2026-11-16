#!/usr/bin/env bash
# Extract the first 40 (sorted) liver volumes + labels from the MSD Task03_Liver tar.
# Usage (from project root, after the download has finished): bash scripts/extract_liver_subset.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export KMP_DUPLICATE_LIB_OK=TRUE

TAR="datasets/Task03_Liver.tar"
OUT="datasets/raw"
N=40

if [ ! -f "$TAR" ]; then
  echo "ERROR: $TAR not found" >&2
  exit 1
fi
mkdir -p "$OUT"

# 1. List image members (skip macOS '._' junk), sort, keep first N.
mapfile -t IMAGES < <(tar -tf "$TAR" \
  | grep -E '(^|/)Task03_Liver/imagesTr/liver_[^/]*\.nii\.gz$' \
  | grep -v '/\._' \
  | sort \
  | head -n "$N")

if [ "${#IMAGES[@]}" -eq 0 ]; then
  echo "ERROR: no imagesTr/liver_*.nii.gz members found in $TAR" >&2
  exit 1
fi

# 2. Matching label members.
LABELS=()
for img in "${IMAGES[@]}"; do
  LABELS+=("$(echo "$img" | sed 's#imagesTr/#labelsTr/#')")
done

echo "Extracting ${#IMAGES[@]} images and ${#LABELS[@]} labels into $OUT ..."
tar -xf "$TAR" -C "$OUT" "${IMAGES[@]}" "${LABELS[@]}"

# 3. Remove macOS junk files.
find "$OUT" -name '._*' -type f -delete

# 4. Report count.
N_IMG=$(find "$OUT/Task03_Liver/imagesTr" -name 'liver_*.nii.gz' 2>/dev/null | wc -l)
N_LBL=$(find "$OUT/Task03_Liver/labelsTr" -name 'liver_*.nii.gz' 2>/dev/null | wc -l)
echo "Extracted: imagesTr=$N_IMG labelsTr=$N_LBL (expected $N each)"

# 5. Delete the 27 GB tar.
rm -f "$TAR"
echo "Deleted $TAR"
