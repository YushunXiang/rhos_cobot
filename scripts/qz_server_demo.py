import base64
import mimetypes
from pathlib import Path

from scripts.qz_vllm_server import request_qwen

def local_image_as_data_url(image_path: Path) -> str:
    """Convert a local image to a data URL for OpenAI-style image_url input."""
    mime_type, _ = mimetypes.guess_type(image_path.name)
    if not mime_type or not mime_type.startswith("image/"):
        mime_type = "image/jpeg"

    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"

def request_with_local_image(image_path):
    # Example: load a local image. Replace with your own path.
    local_image_path = Path(__file__).resolve().parent/image_path
    if not local_image_path.exists():
        raise FileNotFoundError(
            f"Please set local_image_path to an existing image file: {local_image_path}"
        )
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "请先描述图片内容，再给出三个关键词。"},
                {
                    "type": "image_url",
                    "image_url": {"url": local_image_as_data_url(local_image_path)},
                },
            ],
        }
    ]
    request_qwen(messages)

def request_with_web_image(image_url="https://gw.alicdn.com/tfs/TB1_u_nx_v1gK0jSZFsXXcxOVXa-240-60.png"):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "请先描述图片内容，再给出三个关键词。"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": image_url
                    },
                },
            ],
        }
    ]

    request_qwen(messages)

def main():
    image_path = (
        Path(__file__).resolve().parent.parent
        / "examples/piper_real/logs/04221741/model_input_observation/images/cam_high_1.png"
    )
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "请先描述图片内容，再给出三个关键词。"},
                {
                    "type": "image_url",
                    "image_url": {"url": local_image_as_data_url(image_path)},
                },
            ],
        }
    ]

    request_qwen(messages)


if __name__ == "__main__":
    main()
