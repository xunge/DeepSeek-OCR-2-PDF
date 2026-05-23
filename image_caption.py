import base64
import json
import os
import urllib.error
import urllib.request
from pathlib import Path


IMAGE_CAPTION_ENABLED = os.getenv("IMAGE_CAPTION_ENABLED", "false").lower() in (
    "1",
    "true",
    "yes",
    "on",
)
IMAGE_CAPTION_ENDPOINT = os.getenv("IMAGE_CAPTION_ENDPOINT", "").strip()
IMAGE_CAPTION_MODEL = os.getenv("IMAGE_CAPTION_MODEL", "Qwen/Qwen3-VL-2B-Instruct")
IMAGE_CAPTION_PROMPT = os.getenv(
    "IMAGE_CAPTION_PROMPT",
    "Describe this image in detail for a markdown OCR document. Focus on visible objects, text, layout, charts, and any document-relevant information.",
)
IMAGE_CAPTION_TIMEOUT = float(os.getenv("IMAGE_CAPTION_TIMEOUT", "120"))
IMAGE_CAPTION_MAX_TOKENS = int(os.getenv("IMAGE_CAPTION_MAX_TOKENS", "256"))


def captioning_configured():
    return IMAGE_CAPTION_ENABLED and bool(IMAGE_CAPTION_ENDPOINT)


def _image_data_url(image_path):
    image_bytes = Path(image_path).read_bytes()
    encoded = base64.b64encode(image_bytes).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def _parse_caption(response_payload):
    choices = response_payload.get("choices") or []
    if not choices:
        return ""

    message = choices[0].get("message") or {}
    content = message.get("content", "")
    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")).strip())
        return "\n".join(part for part in parts if part).strip()

    return str(content).strip()


def describe_image(image_path):
    if not captioning_configured():
        return ""

    try:
        image_data_url = _image_data_url(image_path)
    except OSError as exc:
        print(f"Image caption skipped for {image_path}: {exc}")
        return ""

    payload = {
        "model": IMAGE_CAPTION_MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": IMAGE_CAPTION_PROMPT},
                    {
                        "type": "image_url",
                        "image_url": {"url": image_data_url},
                    },
                ],
            }
        ],
        "temperature": 0,
        "max_tokens": IMAGE_CAPTION_MAX_TOKENS,
        "stream": False,
    }

    request = urllib.request.Request(
        IMAGE_CAPTION_ENDPOINT,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=IMAGE_CAPTION_TIMEOUT) as response:
            response_payload = json.loads(response.read().decode("utf-8"))
    except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
        print(f"Image caption failed for {image_path}: {exc}")
        return ""

    return _parse_caption(response_payload)


def format_image_description(caption):
    caption = caption.strip()
    if not caption:
        return ""

    if caption.startswith("[IMAGE DESCRIPTION:"):
        return caption

    return f"[IMAGE DESCRIPTION: {caption}]"
