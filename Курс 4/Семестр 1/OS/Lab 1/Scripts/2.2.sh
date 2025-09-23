#!/bin/sh

if [ -z "$1" ]; then
    echo "Usage: $0 process_name"
    exit 1
fi

PROCESS_NAME="$1"

pidin | grep "$PROCESS_NAME"

