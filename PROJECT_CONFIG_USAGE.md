# Project Config Usage Guide

## Overview

The MCP server now supports automatic local file saving with project-specific configuration. Once configured, all generated images will automatically save to your project's local directory.

## How It Works

1. **One-time setup**: Configure your project's output path using `set_project_config()`
2. **Auto-save**: Future image generations automatically save to the configured path
3. **Override**: You can still override the path per-call if needed
4. **Discovery**: The config is stored in `.claude/config.json` and auto-discovered by walking up the directory tree

## Setup

### Step 1: Configure Your Project

Call the `set_project_config` MCP tool once per project:

```python
set_project_config(
    project_root="/path/to/your/project",
    image_output_path="./public/images",  # Relative or absolute path
    project_name="my-app"  # Optional
)
```

This creates: `/path/to/your/project/.claude/config.json`

```json
{
  "image_output_path": "./public/images",
  "project_name": "my-app",
  "last_updated": "2026-01-16T10:30:00Z"
}
```

### Step 2: Generate Images

Now when you generate images, they automatically save locally:

```python
# No output_path needed - uses project config!
generate_image(prompt="cat icon")
# Saves to: /path/to/your/project/public/images/generated_20260116_123456.jpg
```

## MCP Tools

### 1. `set_project_config`

Configure where images should be saved for this project.

**Parameters:**
- `project_root` (str): Path to project root directory
- `image_output_path` (str): Where to save images (relative to project_root or absolute)
- `project_name` (str, optional): Project name for organization

**Returns:**
```json
{
  "status": "success",
  "config_file": "/path/to/project/.claude/config.json",
  "settings": {
    "image_output_path": "./public/images",
    "project_name": "my-app"
  },
  "hint": "Future image generations will automatically save to this path unless overridden"
}
```

### 2. `generate_image` (Updated)

Generate an image using Google Gemini.

**New Parameter:**
- `output_path` (str, optional): Local directory to save image. If not provided, uses project config from `.claude/config.json`

**Returns:**
```json
{
  "url": "https://supabase.url/...",
  "filename": "generated_20260116_123456.jpg",
  "local_path": "/path/to/project/public/images/generated_20260116_123456.jpg",
  "hint": "..."
}
```

### 3. `process_and_store` (Updated)

Process image: remove background, split grid, and store tiles.

**New Parameter:**
- `output_path` (str, optional): Local directory to save tiles. If not provided, uses project config from `.claude/config.json`

**Returns:**
```json
[
  {
    "filename": "icon_home.png",
    "url": "https://supabase.url/...",
    "local_path": "/path/to/project/public/images/icon_home.png"
  },
  ...
]
```

## Usage Examples

### Example 1: First Time Setup

```python
# Step 1: Configure project
set_project_config(
    project_root="/Users/alice/my-app",
    image_output_path="./public/images"
)

# Step 2: Generate images (auto-saves to ./public/images)
generate_image(prompt="logo design")
# Saves to: /Users/alice/my-app/public/images/generated_xxx.jpg
```

### Example 2: Grid Processing

```python
# Generate a 3x3 grid
generate_image(prompt="9 app icons in a 3x3 grid")

# Process and split (auto-saves to configured path)
process_and_store(
    image_source="generated_xxx.jpg",
    names=["home", "settings", "profile", "search", "notifications",
           "messages", "calendar", "help", "logout"]
)
# Each icon saves to: /Users/alice/my-app/public/images/home.png, etc.
```

### Example 3: Override Path

```python
# Use a different path for this specific generation
generate_image(
    prompt="temporary test image",
    output_path="/tmp/test"
)
# Saves to: /tmp/test/generated_xxx.jpg (ignores project config)
```

### Example 4: Multiple Projects

```python
# Project A
set_project_config(
    project_root="/Users/alice/project-a",
    image_output_path="./assets/images"
)

# Project B
set_project_config(
    project_root="/Users/alice/project-b",
    image_output_path="./public/img"
)

# When working in project-a, images save to ./assets/images
# When working in project-b, images save to ./public/img
# Auto-detected by walking up directory tree!
```

## How Config Discovery Works

The server walks up the directory tree (up to 10 levels) looking for `.claude/config.json`:

```
/Users/alice/my-app/src/components/  <- Current directory
/Users/alice/my-app/src/             <- Check here
/Users/alice/my-app/                 <- Found .claude/config.json! ✓
```

This means you can call the MCP functions from any subdirectory of your project, and it will find the config.

## Benefits

✅ **One-time setup**: Configure once per project
✅ **Auto-discovery**: Works from any subdirectory
✅ **Flexible**: Can override per-call if needed
✅ **Persistent**: Config survives across sessions
✅ **Standard**: Uses `.claude/` directory (like `.git/`, `.vscode/`)
✅ **Dual storage**: Still uploads to Supabase as backup

## File Structure

```
your-project/
├── .claude/
│   └── config.json          # Project config (auto-created)
├── public/
│   └── images/              # Generated images (auto-saved here)
│       ├── generated_xxx.jpg
│       ├── icon_home.png
│       └── icon_settings.png
├── src/
│   └── ...
└── package.json
```

## Notes

- The `.claude/` directory should be added to `.gitignore` if you don't want to commit it
- Relative paths in `image_output_path` are resolved relative to `project_root`
- If no config is found, images only upload to Supabase (no local save)
- Local saves happen in addition to Supabase uploads (dual storage)
