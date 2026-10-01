#!/usr/bin/env bash
# Download the data bundle (the GitHub Release rev1-data of this repository), check it and unpack it into data/bundle.
#
#   bash get_data.sh
#
# Needs bash, curl, tar and sha256sum or shasum only (no Python, no git, no Git LFS, no GitHub account). The release
# assets listed in data/RELEASE_ASSETS.sha256 (about 3.1 GB) are downloaded into data/downloads/ (interrupted downloads
# resume), each is checked against its sha256 there, the archives are unpacked into data/bundle/ and every file of the
# bundle is checked against data/MANIFEST.sha256. Running the script again skips what is already present and verified.
# On Windows run it in WSL or Git Bash.
#
# RELEASE_ASSET_DIR=<dir>: copy the assets from a local directory instead of downloading them (offline use).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
BASE_URL="https://github.com/Estebanii/AUKUS-Military-AI-Reproduction/releases/download/rev1-data"
ASSETS=data/RELEASE_ASSETS.sha256
MANIFEST=data/MANIFEST.sha256
DOWNLOADS=data/downloads
BUNDLE=data/bundle

if command -v sha256sum >/dev/null 2>&1; then
  SHA="sha256sum"
elif command -v shasum >/dev/null 2>&1; then
  SHA="shasum -a 256"
else
  echo "get_data.sh: sha256sum or shasum is needed" >&2; exit 2
fi
command -v tar >/dev/null 2>&1 || { echo "get_data.sh: tar is needed" >&2; exit 2; }
if [ -z "${RELEASE_ASSET_DIR:-}" ]; then
  command -v curl >/dev/null 2>&1 || { echo "get_data.sh: curl is needed" >&2; exit 2; }
fi
[ -f "$ASSETS" ] && [ -f "$MANIFEST" ] || { echo "get_data.sh: $ASSETS or $MANIFEST is missing" >&2; exit 2; }
EXPECTED=$(grep -vc '^#' "$MANIFEST")

sha_of() { $SHA "$1" | awk '{print $1}'; }

CHECK_LIST="$(mktemp)"
CHECK_OUT="$(mktemp)"
trap 'rm -f "$CHECK_LIST" "$CHECK_OUT"' EXIT

bundle_ok() {   # data/bundle holds every file of the package's manifest with its sha256 (details in $CHECK_OUT)
  : > "$CHECK_OUT"
  [ -f "$BUNDLE/MANIFEST.sha256" ] && cmp -s "$BUNDLE/MANIFEST.sha256" "$MANIFEST" || return 1
  grep -v '^#' "$MANIFEST" | awk '{print $1 "  " $3}' > "$CHECK_LIST"
  (cd "$BUNDLE" && $SHA -c "$CHECK_LIST") > "$CHECK_OUT" 2>&1
}

if bundle_ok; then
  echo "data/bundle: already present; $EXPECTED files, all sha256 OK"
  exit 0
fi

echo "== release assets -> $DOWNLOADS"
mkdir -p "$DOWNLOADS"
while read -r sum name; do
  [ -n "${name:-}" ] || continue
  file="$DOWNLOADS/$name"
  if [ -f "$file" ] && [ "$(sha_of "$file")" = "$sum" ]; then
    echo "  $name: present, sha256 OK"
    continue
  fi
  if [ -n "${RELEASE_ASSET_DIR:-}" ]; then
    cp "$RELEASE_ASSET_DIR/$name" "$file.part"
  else
    url="$BASE_URL/$name"
    echo "  $name: downloading $url"
    rc=0
    curl -fL --progress-bar --retry 5 --retry-delay 5 -C - -o "$file.part" "$url" </dev/null || rc=$?
    if [ "$rc" -eq 22 ] || [ "$rc" -eq 33 ] || [ "$rc" -eq 36 ]; then   # HTTP error or resume refused: once from the start
      rm -f "$file.part"
      rc=0
      curl -fL --progress-bar --retry 5 --retry-delay 5 -o "$file.part" "$url" </dev/null || rc=$?
    fi
    if [ "$rc" -ne 0 ]; then
      echo "get_data.sh: could not download $url (curl exit status $rc)." >&2
      echo "  The GitHub Release rev1-data must be published and github.com reachable: open the URL in a browser" >&2
      echo "  and check the network connection, then run bash get_data.sh again (verified assets are kept and an" >&2
      echo "  interrupted download resumes)." >&2
      exit 1
    fi
  fi
  got=$(sha_of "$file.part")
  if [ "$got" != "$sum" ]; then
    rm -f "$file.part"
    echo "get_data.sh: $name has sha256 $got, expected $sum; the download was removed, run the script again" >&2
    exit 1
  fi
  mv "$file.part" "$file"
  echo "  $name: sha256 OK"
done < "$ASSETS"

echo "== unpacking -> $BUNDLE"
mkdir -p "$BUNDLE"
for name in $(awk '$2 ~ /\.tar\.gz$/ {print $2}' "$ASSETS"); do
  echo "  $name"
  tar -xzf "$DOWNLOADS/$name" -C "$BUNDLE"
done
cp "$DOWNLOADS/MANIFEST.sha256" "$BUNDLE/MANIFEST.sha256"

echo "== checking every file of the bundle against $MANIFEST"
found=$(grep -vc '^#' "$BUNDLE/MANIFEST.sha256")
if [ "$found" = "$EXPECTED" ] && bundle_ok; then
  echo "data/bundle: $EXPECTED files, all sha256 OK"
  echo "The archives in $DOWNLOADS are no longer needed and may be deleted."
else
  echo "get_data.sh: the bundle in $BUNDLE does not match $MANIFEST; delete $BUNDLE and run the script again" >&2
  grep -v ': OK$' "$CHECK_OUT" >&2 || true
  exit 1
fi
