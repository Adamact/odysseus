"""Image-domain tool implementations.

Extracted from tool_implementations.py as part of slice 1 (#4082/#4071).
Holds the edit_image (gallery) tool.
``src.tool_implementations`` re-exports these for backward compatibility.
``_INTERNAL_BASE`` still lives in tool_implementations.py and is pulled back
function-locally here.
"""
import hashlib
import io
import uuid
from pathlib import Path
from typing import Dict, Optional

from src.tools._common import _parse_tool_args


async def do_edit_image(content: str, owner: Optional[str] = None) -> Dict:
    """Create an owner-scoped edited copy of a gallery image."""
    try:
        args = _parse_tool_args(content)
    except ValueError:
        return {"error": "Invalid JSON arguments", "exit_code": 1}
    image_id = args.get("image_id", "")
    action = args.get("action", "")
    if not image_id or not action:
        return {"error": "image_id and action are required", "exit_code": 1}
    if action not in {"upscale", "rembg"}:
        return {
            "error": f"Unsupported edit action: {action}. Use upscale or rembg.",
            "exit_code": 1,
        }

    from core.database import GalleryImage, SessionLocal
    from src.constants import GENERATED_IMAGES_DIR

    db = SessionLocal()
    try:
        q = db.query(GalleryImage).filter(
            GalleryImage.id == image_id,
            GalleryImage.is_active == True,  # noqa: E712
        )
        # A tool call without an owner must never fall through to another
        # user's gallery row.
        q = q.filter(GalleryImage.owner == owner) if owner else q.filter(False)
        source = q.first()
        if not source:
            return {"error": "Image not found", "exit_code": 1}

        root = Path(GENERATED_IMAGES_DIR).resolve()
        source_name = Path(str(source.filename or "")).name
        source_path = (root / source_name).resolve()
        if source_name != source.filename or source_path.parent != root or not source_path.is_file():
            return {"error": "Image file not found", "exit_code": 1}

        from PIL import Image

        with Image.open(source_path) as opened:
            image = opened.convert("RGBA")
            if action == "upscale":
                try:
                    scale = int(args.get("scale") or 2)
                except (TypeError, ValueError):
                    scale = 2
                if scale not in {2, 4}:
                    return {"error": "scale must be 2 or 4", "exit_code": 1}
                image = image.resize(
                    (image.width * scale, image.height * scale),
                    Image.Resampling.LANCZOS,
                )
            else:
                try:
                    from rembg import remove
                except ImportError:
                    return {
                        "error": "Background removal is not installed. Install the rembg optional dependency.",
                        "exit_code": 1,
                    }
                image = remove(image)

            output = io.BytesIO()
            image.save(output, format="PNG")
            output_bytes = output.getvalue()
            width, height = image.size

        root.mkdir(parents=True, exist_ok=True)
        filename = f"{uuid.uuid4().hex[:12]}.png"
        (root / filename).write_bytes(output_bytes)
        new_id = str(uuid.uuid4())
        derived = GalleryImage(
            id=new_id,
            filename=filename,
            prompt=source.prompt or action,
            caption=source.caption,
            model=f"edit_image:{action}",
            size=f"{width}x{height}",
            quality=source.quality,
            tags=source.tags,
            ai_tags=source.ai_tags,
            session_id=source.session_id,
            album_id=source.album_id,
            owner=owner,
            file_hash=hashlib.sha256(output_bytes).hexdigest(),
            file_size=len(output_bytes),
            width=width,
            height=height,
        )
        db.add(derived)
        db.commit()
        return {
            "output": f"Image edited ({action}). New image ID: {new_id}",
            "exit_code": 0,
            "image_id": new_id,
            "image_url": f"/api/generated-image/{filename}",
            "image_prompt": derived.prompt,
            "image_model": derived.model,
            "image_size": derived.size,
            "image_quality": derived.quality or "",
        }
    except Exception as e:
        db.rollback()
        return {"error": str(e), "exit_code": 1}
    finally:
        db.close()
