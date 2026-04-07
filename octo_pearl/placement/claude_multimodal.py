"""Claude multimodal API: text + image input, text + image output.

Uses Claude Opus 4.6 with vision and the server-side code execution tool so
that Claude can both describe the input image (text output) and generate new
images such as annotated overlays or matplotlib visualisations (image output).
"""

import base64
import os
from pathlib import Path
from typing import Optional

import anthropic


def _encode_image(image_path: str) -> tuple[str, str]:
    """Return (base64_data, media_type) for a local image file."""
    suffix = Path(image_path).suffix.lower()
    media_type = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".webp": "image/webp",
    }.get(suffix, "image/jpeg")

    with open(image_path, "rb") as f:
        data = base64.standard_b64encode(f.read()).decode("utf-8")

    return data, media_type


def claude_multimodal(
    image_path: str,
    text_prompt: str,
    output_dir: str = "./claude_outputs",
    api_key: Optional[str] = None,
) -> tuple[str, list[str]]:
    """Call Claude with text + image input; return text + image output.

    Claude receives the image and prompt, then may write and execute Python
    code (e.g. matplotlib) to produce output images.  Those images are
    downloaded and saved locally.

    Args:
        image_path:  Path to the input image (JPEG / PNG / GIF / WebP).
        text_prompt: Instruction for Claude, e.g. "Annotate the objects in
                     this image and create a labelled bounding-box overlay."
        output_dir:  Directory where generated images are saved.
        api_key:     Anthropic API key.  Falls back to ANTHROPIC_API_KEY env var.

    Returns:
        (text_response, output_image_paths)
        - text_response:       Claude's textual reply.
        - output_image_paths:  Paths to any images Claude generated.
    """
    client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()

    image_data, media_type = _encode_image(image_path)

    # Stream the response — code execution can take time and produce large output.
    with client.messages.stream(
        model="claude-opus-4-6",
        max_tokens=8192,
        thinking={"type": "adaptive"},
        tools=[{"type": "code_execution_20260120", "name": "code_execution"}],
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": image_data,
                        },
                    },
                    {"type": "text", "text": text_prompt},
                ],
            }
        ],
    ) as stream:
        final_message = stream.get_final_message()

    # --- Parse response blocks ---
    text_parts: list[str] = []
    output_image_paths: list[str] = []

    os.makedirs(output_dir, exist_ok=True)

    for block in final_message.content:
        if block.type == "text":
            text_parts.append(block.text)

        elif block.type == "bash_code_execution_tool_result":
            result = block.content
            # The result may be a bash_code_execution_result with nested content
            # (file references) or a plain text stdout/stderr result.
            if not hasattr(result, "content") or not result.content:
                continue

            for file_ref in result.content:
                if file_ref.type != "bash_code_execution_output":
                    continue

                # Fetch metadata then download the generated file.
                metadata = client.beta.files.retrieve_metadata(file_ref.file_id)
                file_content = client.beta.files.download(file_ref.file_id)

                # Sanitise filename to prevent path traversal.
                safe_name = os.path.basename(metadata.filename)
                if not safe_name or safe_name in (".", ".."):
                    print(f"Skipping file with unsafe name: {metadata.filename!r}")
                    continue

                out_path = os.path.join(output_dir, safe_name)
                file_content.write_to_file(out_path)
                output_image_paths.append(out_path)
                print(f"Saved generated image: {out_path}")

    text_response = "\n".join(text_parts)
    return text_response, output_image_paths


# ---------------------------------------------------------------------------
# Quick demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python -m octo_pearl.placement.claude_multimodal <image> <prompt>")
        sys.exit(1)

    img = sys.argv[1]
    prompt = sys.argv[2]

    text, images = claude_multimodal(img, prompt)
    print("\n=== Claude text response ===")
    print(text)
    if images:
        print("\n=== Generated images ===")
        for p in images:
            print(p)
