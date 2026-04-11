#!/usr/bin/env sh

set -eu

BASE_URL="https://ftp.ncbi.nlm.nih.gov/pubmed/baseline/"
WORKDIR="./pubmed_dataset"
ARCHIVE_DIR="$WORKDIR/archives"
XML_DIR="$WORKDIR/xml"

mkdir -p $ARCHIVE_DIR $XML_DIR

echo "Get an list of archive..."
curl -fsSL "$BASE_URL" \
  | grep -oE 'pubmed[0-9]+n[0-9]+\.xml\.gz' \
  | sort -u > "$WORKDIR/filelist.txt"

#if [ ! -s "$WORKDIR/filelist.txt" ]; then
#  echo "Не удалось получить список .xml.gz файлов"
#  exit 1
#fi
#
echo "Download an archive..."
while IFS= read -r file; do
  echo "  -> $file"
  curl -fL --retry 3 -o "$ARCHIVE_DIR/$file" "$BASE_URL/$file"
done < "$WORKDIR/filelist.txt"

echo "Extract an archive ..."
for gz in "$ARCHIVE_DIR"/*.xml.gz; do
  [ -e "$gz" ] || continue
  out="$XML_DIR/$(basename "$gz" .gz)"
  echo "  -> $(basename "$gz")"
  gunzip -c "$gz" > "$out"
done

rm -rf $ARCHIVE_DIR

echo "Successfully finish."

