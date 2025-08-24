"""
merge_side_by_side.py
Merge two 600 × 600 images with a 5‑pixel black spacer in between.

Just edit the three filenames below, then run:
    python merge_side_by_side.py
"""

from pathlib import Path
from PIL import Image  # pip install pillow

# ── ❶ EDIT THESE ────────────────────────────────────────────────────────────────
LEFT_IMAGE   = Path("Current_0614.png")   # first / left‑hand image
RIGHT_IMAGE  = Path("MHRP_0614.png")         # second / right‑hand image
OUTPUT_IMAGE = Path("combined_0614.png")     # where to save the result
# ───────────────────────────────────────────────────────────────────────────────

TARGET_SIZE   = (600, 600)   # size for each source tile (w, h)
DIVIDER_W     = 5            # width of the black vertical bar (px)
DIVIDER_COLOR = (0, 0, 0, 255)  # RGBA black

def merge_images(left_path: Path, right_path: Path, out_path: Path) -> None:
    # 1. open & convert
    left  = Image.open(left_path).convert("RGBA")
    right = Image.open(right_path).convert("RGBA")

    # 2. resize to TARGET_SIZE if needed
    if left.size != TARGET_SIZE:
        left = left.resize(TARGET_SIZE, Image.LANCZOS)
    if right.size != TARGET_SIZE:
        right = right.resize(TARGET_SIZE, Image.LANCZOS)

    # 3. create the canvas: 600 + 5 + 600 = 1205 px wide
    w, h   = TARGET_SIZE
    canvas = Image.new("RGBA", (2 * w + DIVIDER_W, h), (255, 255, 255, 0))

    # 4. paste left tile
    canvas.paste(left, (0, 0))

    # 5. draw (paste) the divider
    divider = Image.new("RGBA", (DIVIDER_W, h), DIVIDER_COLOR)
    canvas.paste(divider, (w, 0))

    # 6. paste right tile (after the divider)
    canvas.paste(right, (w + DIVIDER_W, 0))

    # 7. save
    canvas.save(out_path)
    print(f"Saved → {out_path.resolve()}")

if __name__ == "__main__":
    merge_images(LEFT_IMAGE, RIGHT_IMAGE, OUTPUT_IMAGE)
