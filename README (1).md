# Image Generation MCP Server

MCP server for Google Gemini image generation + processing (split, background removal, Supabase storage).

## Tools

### 1. `generate_image`
Generate images using Google Gemini API.

```python
generate_image(
    prompt: str,                    # Image generation prompt
    model: str = "gemini-2.0-flash-preview-image-generation",
    aspect_ratio: str = "1:1"       # "1:1", "16:9", "4:3", etc.
)
# Returns: {url, filename, hint}
```

### 2. `process_and_store`
Split grid images, remove backgrounds, store to Supabase.

```python
process_and_store(
    image_url: str,      # URL from generate_image or any URL
    names: list[str],    # ["apple", "star", "moon"] - length = perfect square
    remove_bg: bool = True  # True → PNG, False → JPG
)
# Returns: [{filename, url}, ...]
```

**Grid size inferred from `len(names)`:**
- 1 name → no split
- 4 names → 2×2
- 9 names → 3×3
- 16 names → 4×4

## Setup

### 1. Install Dependencies

```bash
pip install "mcp[cli]" httpx Pillow supabase "rembg[cli]"
```

### 2. Environment Variables

```bash
export GOOGLE_API_KEY="your-google-api-key"
export SUPABASE_URL="https://xxx.supabase.co"
export SUPABASE_SECRET_KEY="your-service-role-key"
```

### 3. Supabase Setup

1. Create a **public** bucket named `files`
2. Enable public access in bucket settings
3. Create folder `mcp-images/` (optional, created automatically)

### 4. Add to Claude Code

```bash
claude mcp add --transport stdio image-gen \
  --env GOOGLE_API_KEY=your-key \
  --env SUPABASE_URL=https://xxx.supabase.co \
  --env SUPABASE_SECRET_KEY=your-key \
  -- python /path/to/server.py
```

Or with uv:

```bash
claude mcp add --transport stdio image-gen \
  --env GOOGLE_API_KEY=your-key \
  --env SUPABASE_URL=https://xxx.supabase.co \
  --env SUPABASE_SECRET_KEY=your-key \
  -- uv run /path/to/server.py
```

## Example Usage

**Generate a single image:**
```
generate_image("A cute cat wearing a hat")
→ {url: "https://...", filename: "generated_20250113_120000.jpg"}
```

**Generate grid + process:**
```
generate_image("Nine whimsical icons in a 3x3 layout on white background")
→ {url: "https://...", hint: "Consider using process_and_store..."}

process_and_store(
    image_url="https://...",
    names=["star", "moon", "sun", "cloud", "rain", "snow", "wind", "lightning", "rainbow"],
    remove_bg=True
)
→ [
    {filename: "star.png", url: "https://..."},
    {filename: "moon.png", url: "https://..."},
    ...
  ]
```

**Without background removal:**
```
process_and_store(
    image_url="https://...",
    names=["photo1", "photo2", "photo3", "photo4"],
    remove_bg=False
)
→ [{filename: "photo1.jpg", url: "https://..."}, ...]
```

## Notes

- Background removal uses `birefnet-general` model (first call downloads ~180MB)
- Supabase bucket must be public for returned URLs to work
- Grid splitting assumes equal-sized tiles
- Names are sanitized for filenames (spaces → underscores)
