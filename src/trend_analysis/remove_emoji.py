#!/usr/bin/env python3
import argparse
import io
import os
import re
import sys

# Regex covering common emoji ranges (Emoticons, Misc Symbols & Pictographs, Transport & Map, Supplemental Symbols & Pictographs,
# Symbols & Pictographs Extended-A, Dingbats, Symbols & Dingbats, Flags, Regional Indicators, etc.)
EMOJI_PATTERN = re.compile(
    "["

    "\U0001F300-\U0001F5FF"  # Misc Symbols and Pictographs
    "\U0001F600-\U0001F64F"  # Emoticons
    "\U0001F680-\U0001F6FF"  # Transport & Map
    "\U0001F700-\U0001F77F"  # Alchemical Symbols
    "\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
    "\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
    "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
    "\U0001FA00-\U0001FA6F"  # Chess Symbols + Symbols & Pictographs Extended-A (part)
    "\U0001FA70-\U0001FAFF"  # Symbols & Pictographs Extended-A (rest)
    "\U00002700-\U000027BF"  # Dingbats
    "\U00002600-\U000026FF"  # Misc symbols
    "\U00002500-\U000025FF"  # Box Drawing/Geometric-ish (some terminals treat icons)
    "\U00002300-\U000023FF"  # Misc Technical
    "\U00002B00-\U00002BFF"  # Misc Symbols and Arrows
    "\U0001F1E6-\U0001F1FF"  # Regional Indicator Symbols (flags)
    "\U0001F201-\U0001F2FF"  # Enclosed Ideographic Supplement (part)
    "\U0001F3FB-\U0001F3FF"  # Emoji modifiers (skin tones)
    "]",
    flags=re.UNICODE
)

# Variation Selector-16 (VS16) used to force emoji presentation; remove to avoid stray selectors
VS16 = "\uFE0F"
VS15 = "\uFE0E"

def strip_emojis(text: str) -> str:
    # Remove variation selectors first, then emoji ranges
    text = text.replace(VS16, "").replace(VS15, "")
    text = EMOJI_PATTERN.sub("", text)
    return text

def main():
    parser = argparse.ArgumentParser(description="Remove emojis from a Python source file.")
    parser.add_argument("input_file", help="Path to the Python file to process")
    parser.add_argument("--out", "-o", help="Optional output file (default: overwrite input)")
    parser.add_argument("--dry-run", action="store_true", help="Print cleaned content to stdout without writing")
    args = parser.parse_args()

    in_path = args.input_file
    if not os.path.isfile(in_path):
        print(f"Error: File not found: {in_path}", file=sys.stderr)
        sys.exit(1)

    # Read the file as UTF-8; if BOM exists, Python handles it.
    try:
        with io.open(in_path, "r", encoding="utf-8") as f:
            original = f.read()
    except UnicodeDecodeError:
        # Fallback: try system default if not UTF-8
        with io.open(in_path, "r", encoding=sys.getdefaultencoding(), errors="replace") as f:
            original = f.read()

    cleaned = strip_emojis(original)

    if args.dry_run:
        # Write to stdout using a safe encoding for the console
        sys.stdout.reconfigure(encoding="utf-8", errors="replace") if hasattr(sys.stdout, "reconfigure") else None
        sys.stdout.write(cleaned)
        return

    out_path = args.out or in_path

    # Only write if content changed
    if cleaned == original and out_path == in_path:
        print("No emojis found; file unchanged.")
        return

    # Write back as UTF-8 to preserve non-emoji Unicode safely
    with io.open(out_path, "w", encoding="utf-8", newline="") as f:
        f.write(cleaned)

    if out_path == in_path:
        print(f"Emojis removed. Updated file: {out_path}")
    else:
        print(f"Emojis removed. Wrote cleaned file: {out_path}")

if __name__ == "__main__":
    main()
