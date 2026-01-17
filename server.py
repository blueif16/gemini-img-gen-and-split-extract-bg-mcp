"""
MCP Server for Google Image Generation + Processing
Tools:
  1. generate_image - Generate images using Google Gemini
  2. process_and_store - Split grid, remove bg, store to Supabase
"""

import os
import sys
import math
import base64
import asyncio
from io import BytesIO
from pathlib import Path
from datetime import datetime
from typing import Optional, Any, Tuple
from contextvars import ContextVar

# ============================================================
# Load Environment Variables
# ============================================================

from dotenv import load_dotenv
load_dotenv()

# ============================================================
# Dependency Check
# ============================================================

REQUIRED_PACKAGES = {
    "mcp": "mcp[cli]",
    "httpx": "httpx",
    "PIL": "Pillow",
    "supabase": "supabase",
    "transformers": "transformers",
    "torch": "torch",
}

def check_dependencies():
    """Check and prompt for missing dependencies"""
    missing = []
    for module, package in REQUIRED_PACKAGES.items():
        try:
            __import__(module)
        except ImportError:
            missing.append(package)
    
    if missing:
        print("❌ Missing dependencies detected!", file=sys.stderr)
        print(f"Please install: pip install {' '.join(missing)}", file=sys.stderr)
        print("\nOr run:", file=sys.stderr)
        print(f"  pip install {' '.join(missing)}", file=sys.stderr)
        sys.exit(1)

check_dependencies()

# ============================================================
# Imports (after dependency check)
# ============================================================

import json
import httpx
import torch
from PIL import Image
from supabase import create_client
from mcp.server.fastmcp import FastMCP
from transformers import pipeline

# ============================================================
# Configuration
# ============================================================

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_SECRET_KEY = os.environ.get("SUPABASE_SECRET_KEY")

if not GOOGLE_API_KEY:
    print("❌ GOOGLE_API_KEY environment variable not set", file=sys.stderr)
    sys.exit(1)

if not SUPABASE_URL or not SUPABASE_SECRET_KEY:
    print("❌ SUPABASE_URL and SUPABASE_SECRET_KEY environment variables required", file=sys.stderr)
    sys.exit(1)

# Default settings
DEFAULT_MODEL = "gemini-3-pro-image-preview"
DEFAULT_BG_REMOVAL_MODEL = "briaai/RMBG-1.4"
SUPABASE_BUCKET = "files"
BASE_DIR = "output"
USER_ID = os.environ.get("USER_ID", "550e8400-e29b-41d4-a716-446655440000")

# Context var for current user
current_user_id: ContextVar[Optional[str]] = ContextVar(
    'current_user_id', default=None
)

# Detect device for GPU acceleration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================================
# Initialize MCP Server
# ============================================================

mcp = FastMCP("image-gen")

# ============================================================
# Supabase Client
# ============================================================

_supabase_client = None
_supabase_initialized = False

def get_supabase():
    global _supabase_client, _supabase_initialized
    if _supabase_initialized:
        return _supabase_client
    try:
        if SUPABASE_URL and SUPABASE_SECRET_KEY:
            _supabase_client = create_client(SUPABASE_URL, SUPABASE_SECRET_KEY)
        _supabase_initialized = True
    except Exception as e:
        print(f"⚠️ Supabase initialization failed: {e}", file=sys.stderr)
    return _supabase_client

# ============================================================
# Background Removal Pipeline (lazy loaded)
# ============================================================

_bg_removal_pipe = None

def get_bg_removal_pipeline():
    global _bg_removal_pipe
    if _bg_removal_pipe is None:
        print(f"🔄 Loading background removal model: {DEFAULT_BG_REMOVAL_MODEL} on {DEVICE}", file=sys.stderr)
        _bg_removal_pipe = pipeline(
            "image-segmentation",
            model=DEFAULT_BG_REMOVAL_MODEL,
            trust_remote_code=True,
            device=DEVICE
        )
        print("✅ Background removal model loaded", file=sys.stderr)
    return _bg_removal_pipe

# ============================================================
# Helper Functions
# ============================================================

async def call_gemini_api(
    prompt: str,
    model: str = DEFAULT_MODEL,
    aspect_ratio: Optional[str] = None
) -> dict:
    """Call Google Gemini API for image generation"""
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={GOOGLE_API_KEY}"
    
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "responseModalities": ["TEXT", "IMAGE"]
        }
    }
    
    if aspect_ratio:
        payload["generationConfig"]["imageConfig"] = {"aspectRatio": aspect_ratio}
    
    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.post(url, json=payload)
        if resp.status_code != 200:
            raise RuntimeError(f"Gemini API error {resp.status_code}: {resp.text[:300]}")
        return resp.json()


def extract_image_from_response(response: dict) -> Optional[bytes]:
    """Extract image bytes from Gemini API response"""
    candidates = response.get("candidates", [])
    if not candidates:
        return None
    
    parts = candidates[0].get("content", {}).get("parts", [])
    for part in parts:
        inline_data = part.get("inline_data") or part.get("inlineData")
        if inline_data and inline_data.get("data"):
            return base64.b64decode(inline_data["data"])
    return None


async def upload_to_supabase(
    image_data: bytes,
    filename: str,
    content_type: str = "image/png"
) -> str:
    """Upload image to Supabase and return public URL (legacy wrapper)"""
    result = await save_image_to_supabase(
        image_data=image_data,
        filename=filename,
        save_local=True,
        content_type=content_type
    )
    if result:
        return result[1]  # Return the URL
    raise RuntimeError("Failed to upload to Supabase")


def ensure_folder_structure(folder_type: str) -> Path:
    """确保文件夹结构存在

    Args:
        folder_type: 文件夹类型 ("image", "video", "audio" 等)

    Returns:
        Path: 创建的文件夹路径
    """
    base = Path(__file__).parent / BASE_DIR
    folder = base / f"{folder_type}s"
    folder.mkdir(parents=True, exist_ok=True)
    return folder


async def save_image_to_supabase(
    image_data: bytes,
    filename: str,
    save_local: bool = True,
    content_type: str = "image/png"
) -> Optional[Tuple[str, str]]:
    """上传图片到Supabase bucket，可选保存本地副本
    
    Args:
        image_data: 图片二进制数据
        filename: 文件名
        save_local: 是否保存本地副本
        content_type: MIME类型
    
    Returns:
        Optional[Tuple[str, str]]: (storage_path, public_url) 或 None
    """
    user_id = current_user_id.get() or USER_ID
    storage_path = f"{user_id}/images/{filename}"

    # 保存本地副本（无论Supabase是否可用）
    local_path = None
    if save_local:
        local_folder = ensure_folder_structure("image")
        local_path = local_folder / filename
        with open(local_path, "wb") as f:
            f.write(image_data)
        print(f"💾 本地保存: {local_path}", file=sys.stderr)

    supabase_client = get_supabase()
    if not supabase_client:
        print("⚠️  Supabase不可用，仅保存本地副本", file=sys.stderr)
        return (storage_path, str(local_path)) if local_path else None

    try:
        print(f"☁️  上传: {storage_path}", file=sys.stderr)

        try:
            supabase_client.storage.from_(SUPABASE_BUCKET).upload(
                storage_path, image_data,
                file_options={"content-type": content_type, "cache-control": "3600"}
            )
            print("✅ 上传成功", file=sys.stderr)
        except Exception as upload_error:
            error_str = str(upload_error)
            if ("409" in error_str or "Duplicate" in error_str or
                    "already exists" in error_str):
                print("⚠️  文件已存在，更新...", file=sys.stderr)
                supabase_client.storage.from_(SUPABASE_BUCKET).update(
                    storage_path, image_data,
                    file_options={"content-type": content_type, "cache-control": "3600"}
                )
                print("✅ 更新成功", file=sys.stderr)
            else:
                raise

        # 更新或插入元数据
        existing = (
            supabase_client.table("user_images")
            .select("id")
            .eq("user_id", user_id)
            .eq("storage_path", storage_path)
            .execute()
        )

        if existing.data:
            supabase_client.table("user_images").update({
                "filename": filename,
                "file_size": len(image_data),
                "mime_type": content_type
            }).eq("user_id", user_id).eq(
                "storage_path", storage_path
            ).execute()
        else:
            supabase_client.table("user_images").insert({
                "user_id": user_id,
                "storage_path": storage_path,
                "filename": filename,
                "file_size": len(image_data),
                "mime_type": content_type
            }).execute()

        # 生成签名URL
        signed_url_response = (
            supabase_client.storage.from_(SUPABASE_BUCKET)
            .create_signed_url(storage_path, expires_in=3600)
        )

        public_url = None
        if isinstance(signed_url_response, dict):
            public_url = (
                signed_url_response.get("signedURL") or
                signed_url_response.get("signedUrl") or
                signed_url_response.get("signed_url") or
                signed_url_response.get("url")
            )
        elif isinstance(signed_url_response, str):
            public_url = signed_url_response

        return (storage_path, public_url)
    except Exception as e:
        print(f"❌ 上传异常: {type(e).__name__}: {str(e)}", file=sys.stderr)
        return None


async def get_image_url_from_asset_id(asset_id: str) -> Optional[str]:
    """从asset_id获取图片URL"""
    supabase_client = get_supabase()
    if not supabase_client:
        raise RuntimeError("Supabase未配置")
    try:
        response = (
            supabase_client.table("user_images")
            .select("storage_path")
            .eq("id", asset_id)
            .single()
            .execute()
        )
        if not response.data:
            raise Exception("文件不存在")
        storage_path = response.data.get("storage_path")

        signed_url_response = (
            supabase_client.storage.from_(SUPABASE_BUCKET)
            .create_signed_url(storage_path, expires_in=3600)
        )

        if isinstance(signed_url_response, dict):
            signed_url = (
                signed_url_response.get("signedURL") or
                signed_url_response.get("signedUrl")
            )
            if signed_url:
                return signed_url
        elif isinstance(signed_url_response, str):
            return signed_url_response

        raise Exception(f"生成URL失败: {type(signed_url_response)}")
    except Exception as e:
        print(f"❌ 获取URL异常: {type(e).__name__}: {str(e)}", file=sys.stderr)
        return None


async def save_to_supabase(
    file_path: Path,
    folder_type: str
) -> Optional[Tuple[str, str]]:
    """上传文件到Supabase并保存元数据

    Args:
        file_path: 本地文件路径
        folder_type: 文件夹类型 ("image", "video", "audio" 等)

    Returns:
        Optional[Tuple[str, str]]: (storage_path, public_url) 或 None
    """
    if not file_path.exists():
        print(f"❌ 文件不存在: {file_path}", file=sys.stderr)
        return None

    with open(file_path, "rb") as f:
        file_data = f.read()

    filename = file_path.name
    
    # 根据扩展名确定content_type
    ext = file_path.suffix.lower()
    content_type_map = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif": "image/gif",
        ".webp": "image/webp",
    }
    content_type = content_type_map.get(ext, "application/octet-stream")
    
    return await save_image_to_supabase(
        image_data=file_data,
        filename=filename,
        save_local=False,  # 已经有本地文件了
        content_type=content_type
    )


def split_image_grid(img: Image.Image, rows: int, cols: int) -> list[Image.Image]:
    """Split image into grid of tiles"""
    w, h = img.size
    tile_w, tile_h = w // cols, h // rows
    
    tiles = []
    for i in range(rows):
        for j in range(cols):
            box = (j * tile_w, i * tile_h, (j + 1) * tile_w, (i + 1) * tile_h)
            tiles.append(img.crop(box))
    return tiles


def remove_background(img: Image.Image) -> Image.Image:
    """Remove background from image using RMBG-1.4 model"""
    pipe = get_bg_removal_pipeline()
    # RMBG-1.4 returns the image with background removed directly
    result = pipe(img)
    return result.convert("RGBA")


# ============================================================
# Project Config Management
# ============================================================

def find_project_config(start_path: Optional[str] = None) -> Optional[Path]:
    """Walk up directory tree to find .claude/config.json

    Args:
        start_path: Starting directory (defaults to cwd)

    Returns:
        Path to config file or None if not found
    """
    current = Path(start_path or os.getcwd()).resolve()

    # Check up to 10 levels up
    for _ in range(10):
        config_file = current / ".claude" / "config.json"
        if config_file.exists():
            return config_file

        parent = current.parent
        if parent == current:  # Reached root
            break
        current = parent

    return None


def get_project_config(start_path: Optional[str] = None) -> Optional[dict]:
    """Load project configuration from .claude/config.json

    Args:
        start_path: Starting directory (defaults to cwd)

    Returns:
        Config dict or None if not found
    """
    config_file = find_project_config(start_path)
    if not config_file:
        return None

    try:
        with open(config_file, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"⚠️ Failed to load config: {e}", file=sys.stderr)
        return None


def save_project_config(config_path: str, config_data: dict) -> Path:
    """Save project configuration to .claude/config.json

    Args:
        config_path: Path to project root
        config_data: Configuration data to save

    Returns:
        Path to saved config file
    """
    config_dir = Path(config_path).resolve() / ".claude"
    config_dir.mkdir(parents=True, exist_ok=True)

    config_file = config_dir / "config.json"

    # Merge with existing config if present
    existing = {}
    if config_file.exists():
        try:
            with open(config_file, "r") as f:
                existing = json.load(f)
        except Exception:
            pass

    existing.update(config_data)
    existing["last_updated"] = datetime.now().isoformat()

    with open(config_file, "w") as f:
        json.dump(existing, f, indent=2)

    return config_file


def save_image_locally(image_data: bytes, filename: str, output_path: str) -> Path:
    """Save image to user-specified local path

    Args:
        image_data: Image binary data
        filename: Filename to save as
        output_path: Directory path to save to

    Returns:
        Path to saved file
    """
    output_dir = Path(output_path).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    file_path = output_dir / filename
    with open(file_path, "wb") as f:
        f.write(image_data)

    return file_path


# ============================================================
# MCP Tools
# ============================================================

@mcp.tool()
async def set_project_config(
    project_root: str,
    image_output_path: str,
    project_name: Optional[str] = None
) -> dict:
    """Configure where images should be saved for this project.

    Creates .claude/config.json in the project root to remember settings.
    Future image generations will automatically use this path.

    Args:
        project_root: Path to project root directory
        image_output_path: Where to save images (relative to project_root or absolute)
        project_name: Optional project name for organization

    Returns:
        Confirmation with saved configuration
    """
    config_data = {
        "image_output_path": image_output_path,
    }

    if project_name:
        config_data["project_name"] = project_name

    config_file = save_project_config(project_root, config_data)

    print(f"✅ Project config saved to: {config_file}", file=sys.stderr)

    return {
        "status": "success",
        "config_file": str(config_file),
        "settings": config_data,
        "hint": "Future image generations will automatically save to this path unless overridden"
    }


@mcp.tool()
async def generate_image(
    prompt: str,
    model: str = DEFAULT_MODEL,
    aspect_ratio: str = "1:1",
    output_path: Optional[str] = None
) -> dict:
    """Generate an image using Google Gemini.

    Args:
        prompt: Image generation prompt. Can include layout hints like "3x3 grid" or "nine icons".
        model: Gemini model to use. Default: gemini-2.0-flash-preview-image-generation
        aspect_ratio: Image aspect ratio (e.g., "1:1", "16:9", "4:3"). Default: 1:1
        output_path: Local directory to save image. If not provided, uses project config from .claude/config.json

    Returns:
        dict with 'url' (Supabase), 'filename', 'local_path' (if saved locally), and 'hint' for follow-up actions
    """
    print(f"🎨 Generating image with prompt: {prompt[:100]}...", file=sys.stderr)

    # Call Gemini API
    response = await call_gemini_api(prompt, model, aspect_ratio)

    # Extract image
    image_data = extract_image_from_response(response)
    if not image_data:
        raise RuntimeError("No image data in response")

    # Generate filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    temp_filename = f"generated_{timestamp}.jpg"

    # Upload to Supabase
    url = await upload_to_supabase(image_data, temp_filename, "image/jpeg")

    result = {
        "url": url,
        "filename": temp_filename,
    }

    # Try to get output path from: parameter > project config
    if output_path is None:
        config = get_project_config()
        if config:
            output_path = config.get("image_output_path")
            print(f"📋 Using output path from project config: {output_path}", file=sys.stderr)

    # Save locally if we have a path
    if output_path:
        local_file = save_image_locally(image_data, temp_filename, output_path)
        result["local_path"] = str(local_file)
        print(f"💾 Saved locally: {local_file}", file=sys.stderr)

    print(f"✅ Image generated and uploaded: {temp_filename}", file=sys.stderr)

    # Check if prompt suggests grid layout
    hint = None
    grid_keywords = ["grid", "layout", "arranged", "icons", "tiles", "2x2", "3x3", "4x4",
                     "four ", "nine ", "sixteen ", "4 ", "9 ", "16 "]
    prompt_lower = prompt.lower()
    if any(kw in prompt_lower for kw in grid_keywords):
        hint = ("This prompt suggests a grid layout. Consider using process_and_store() "
                "with a list of names to split the image, optionally remove backgrounds, "
                "and store each tile with a proper filename.")

    if hint:
        result["hint"] = hint

    return result


@mcp.tool()
async def process_and_store(
    image_source: str,
    names: list[str],
    remove_bg: bool = True,
    output_path: Optional[str] = None
) -> list[dict]:
    """Process image: remove background first (1 API call), then split grid and store tiles to Supabase.

    Args:
        image_source: URL or local file path of the image to process
        names: List of names for each tile. Length determines grid size (must be perfect square: 1, 4, 9, 16...)
        remove_bg: Whether to remove background first. Default: True (outputs PNG). False outputs JPG.
        output_path: Local directory to save tiles. If not provided, uses project config from .claude/config.json

    Returns:
        List of dicts with 'filename', 'url', and 'local_path' (if saved locally) for each processed tile

    Note:
        Extract-first approach: 1 API call for entire image, then split locally (9x faster, 9x cheaper)
    """
    count = len(names)

    # Validate perfect square
    grid_size = int(math.sqrt(count))
    if grid_size * grid_size != count:
        raise ValueError(f"names count ({count}) must be a perfect square (1, 4, 9, 16, ...)")

    print(f"🔄 Processing image: {grid_size}x{grid_size} grid, remove_bg={remove_bg}", file=sys.stderr)

    # Try to get output path from: parameter > project config
    if output_path is None:
        config = get_project_config()
        if config:
            output_path = config.get("image_output_path")
            print(f"📋 Using output path from project config: {output_path}", file=sys.stderr)

    # Load image from URL or local file
    if image_source.startswith(("http://", "https://")):
        # Download from URL
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.get(image_source)
            resp.raise_for_status()
            img = Image.open(BytesIO(resp.content)).convert("RGBA")
    else:
        # Load from local file
        if not os.path.exists(image_source):
            raise FileNotFoundError(f"Local file not found: {image_source}")
        img = Image.open(image_source).convert("RGBA")
        print(f"📁 Loaded local file: {image_source}", file=sys.stderr)

    # STEP 1: Remove background FIRST (1 API call for entire image)
    if remove_bg:
        print(f"🔄 Removing background from entire image (1 API call)...", file=sys.stderr)
        img = remove_background(img)
        ext = "png"
        content_type = "image/png"
        print(f"✅ Background removed from entire image", file=sys.stderr)
    else:
        img = img.convert("RGB")
        ext = "jpg"
        content_type = "image/jpeg"

    # STEP 2: Split into tiles (local operation, no API calls)
    if count == 1:
        tiles = [img]
    else:
        tiles = split_image_grid(img, grid_size, grid_size)

    print(f"📦 Split into {len(tiles)} tiles", file=sys.stderr)

    # STEP 3: Upload each tile
    results = []
    for i, (tile, name) in enumerate(zip(tiles, names)):
        # Clean name for filename
        clean_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)
        filename = f"{clean_name}.{ext}"

        # Convert to bytes
        buffer = BytesIO()
        if ext == "png":
            tile.save(buffer, format="PNG")
        else:
            tile.save(buffer, format="JPEG", quality=95)
        image_bytes = buffer.getvalue()

        # Upload to Supabase
        url = await upload_to_supabase(image_bytes, filename, content_type)

        result = {
            "filename": filename,
            "url": url
        }

        # Save locally if we have a path
        if output_path:
            local_file = save_image_locally(image_bytes, filename, output_path)
            result["local_path"] = str(local_file)

        print(f"✅ [{i+1}/{count}] Uploaded: {filename}", file=sys.stderr)

        results.append(result)

    return results


# ============================================================
# Main
# ============================================================

def main():
    print("🚀 Starting Image Gen MCP Server...", file=sys.stderr)
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
