#!/bin/sh

if [ -z "$1" ]; then
    echo "Usage: $0 source_file.c"
    exit 1
fi

SOURCE_FILE="$1"
OUTPUT_FILE="${SOURCE_FILE%.*}"

gcc "$SOURCE_FILE" -o "$OUTPUT_FILE"
if [ $? -eq 0 ]; then
    ./"$OUTPUT_FILE"
else
    vi "$SOURCE_FILE"
fi
