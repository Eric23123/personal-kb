import sys as _sys
from pathlib import Path as _Path
_sys_root = _Path(__file__).resolve().parents[2]
if str(_sys_root) not in _sys.path:
    _sys.path.insert(0, str(_sys_root))


import argparse
import base64
import hashlib
import io
import json
import os
import tempfile

"""
Personal KB — Diagram Extraction Script

Extracts images/diagrams from PDFs and generates structured descriptions
using Qwen-VL through DashScope. Descriptions are stored as searchable facts.

Usage:
    python diagrams.py lecture_slides.pdf --course "PERSONAL-ALPHA" --lecture 1
    python diagrams.py lecture_slides.pdf --output diagram_facts.json
"""

try:
    from ..core.qwen_vision import DEFAULT_QWEN_VISION_API_URL, DEFAULT_QWEN_VISION_MODEL, QwenVisionClient
except ImportError:  # pragma: no cover - direct script execution
    from scripts.core.qwen_vision import DEFAULT_QWEN_VISION_API_URL, DEFAULT_QWEN_VISION_MODEL, QwenVisionClient

QWEN_VISION_API_URL = os.environ.get("PERSONAL_KB_QWEN_VISION_API_URL", DEFAULT_QWEN_VISION_API_URL)
VISION_MODEL = os.environ.get("PERSONAL_KB_QWEN_VISION_MODEL", DEFAULT_QWEN_VISION_MODEL)

DIAGRAM_PROMPTS = {
    "block_diagram": "Describe this block diagram in detail: identify all blocks/components with their labels, all connections and arrows showing signal flow, any equations or transfer functions shown, and the overall system being represented.",
    "flowchart": "Describe this flowchart: identify each step, decision point, arrows showing flow direction, start/end points, and any annotations or labels.",
    "circuit": "Describe this circuit diagram: identify all components (resistors, capacitors, op-amps, etc.), their values, connections between them, and the overall circuit function.",
    "graph": "Describe this graph/plot: identify the axes labels and units, all data series with their labels, trends observed, key data points, and any equations or annotations.",
    "equation": "Read all equations and formulas shown in this image. Output them in standard mathematical notation.",
    "matlab": "Describe this MATLAB/Simulink output: identify the system being simulated, parameters shown, key observations from the plots.",
    "general": "Describe this technical diagram in detail: what type of diagram is it, what are the main components, what relationships are shown, what labels/values/equations appear? Include all text visible in the image.",
}


def _qwen_transport(image_b64, prompt, api_url, model, timeout):
    """Send one Qwen-VL request; injectable alternatives keep tests offline."""
    return QwenVisionClient(api_url=api_url, timeout=timeout).complete_image(prompt, image_b64, model=model)


def _encode_image(image_path, max_dim=2048):
    """Return an image as base64, resizing in memory instead of creating temp files."""
    from PIL import Image

    with Image.open(image_path) as img:
        if img.width <= max_dim and img.height <= max_dim:
            with open(image_path, "rb") as f:
                return base64.b64encode(f.read()).decode()
        ratio = min(max_dim / img.width, max_dim / img.height)
        resized = img.resize((int(img.width * ratio), int(img.height * ratio)), Image.LANCZOS)
        buffer = io.BytesIO()
        resized.save(buffer, "PNG")
        return base64.b64encode(buffer.getvalue()).decode()


def _parse_diagram_response(response, fallback_type="general"):
    """Parse the structured response, while tolerating models that add prose/code fences."""
    text = (response or "").strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start >= 0 and end > start:
            try:
                parsed = json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                parsed = {}
        else:
            parsed = {}
    diagram_type = parsed.get("diagram_type", fallback_type)
    if diagram_type not in DIAGRAM_PROMPTS:
        diagram_type = fallback_type if fallback_type in DIAGRAM_PROMPTS else "general"
    return str(parsed.get("description") or text), diagram_type


def analyze_diagram(image_path, api_url=QWEN_VISION_API_URL, model=VISION_MODEL, transport=None):
    """Classify and describe a diagram in a single vision-model request.

    ``transport`` receives ``(image_b64, prompt, api_url, model, timeout)`` and
    returns text, making this path deterministic and network-free in tests.
    """
    prompt = (
        "Analyze this technical image once. Reply with ONLY JSON in this exact "
        "shape: {\"diagram_type\": \"one of block_diagram, flowchart, circuit, "
        "graph, equation, matlab, general\", \"description\": \"detailed "
        "description including visible text, labels, values, equations, components, "
        "and relationships\"}."
    )
    image_b64 = _encode_image(image_path)
    response = (transport or _qwen_transport)(image_b64, prompt, api_url, model, 180)
    return _parse_diagram_response(response)


def detect_diagram_type(image_path, api_url=QWEN_VISION_API_URL, model=VISION_MODEL, transport=None):
    """Compatibility helper; uses the same one-pass structured vision response."""
    _, diagram_type = analyze_diagram(image_path, api_url, model, transport)
    return diagram_type


def describe_diagram(image_path, diagram_type=None, api_url=QWEN_VISION_API_URL, model=VISION_MODEL, transport=None):
    """Generate a description and type in one vision call (``diagram_type`` is legacy)."""
    description, detected_type = analyze_diagram(image_path, api_url, model, transport)
    return description, detected_type if diagram_type is None else diagram_type


def deduplicate_extracted_images(images):
    """Keep the first extracted image for each identical byte sequence."""
    unique_images = []
    fingerprints = set()
    for image in images:
        try:
            with open(image[2], "rb") as f:
                fingerprint = hashlib.sha256(f.read()).digest()
        except OSError:
            # Let the later processing step report an unavailable file normally.
            unique_images.append(image)
            continue
        if fingerprint not in fingerprints:
            fingerprints.add(fingerprint)
            unique_images.append(image)
    return unique_images


def extract_images_from_pdf(pdf_path, output_dir=None):
    """Extract all images from a PDF. Returns list of (page_num, img_index, img_path)."""
    import pymupdf

    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="personal_kb_diagrams_")

    doc = pymupdf.open(pdf_path)
    images = []
    try:
        for page_num in range(len(doc)):
            page = doc[page_num]

            # A scanned page is represented by its full-page render only. This
            # prevents the embedded scan and the render from being ingested twice.
            text_dict = page.get_text("dict")
            text_blocks = [b for b in text_dict["blocks"] if b.get("type") == 0 and b.get("text", "").strip()]
            image_blocks = [b for b in text_dict["blocks"] if b.get("type") == 1]
            page_images = page.get_images(full=True)
            if not text_blocks and (page_images or image_blocks):
                pix = page.get_pixmap(dpi=150)
                img_path = os.path.join(output_dir, f"page{page_num+1}_full.png")
                pix.save(img_path)
                images.append((page_num + 1, 0, img_path))
                continue

            for img_index, img in enumerate(page_images):
                xref = img[0]
                try:
                    pix = pymupdf.Pixmap(doc, xref)
                    if pix.width < 100 or pix.height < 100:
                        continue
                    img_path = os.path.join(output_dir, f"page{page_num+1}_img{img_index+1}.png")
                    pix.save(img_path)
                    images.append((page_num + 1, img_index + 1, img_path))
                except Exception as e:
                    print(f"  Warning: couldn't extract image {img_index+1} from page {page_num+1}: {e}")
    finally:
        doc.close()
    return deduplicate_extracted_images(images)


def process_pdf_diagrams(
    pdf_path, course_code, lecture, api_url=QWEN_VISION_API_URL, model=VISION_MODEL,
    extractor=None, transport=None, temp_root=None,
):
    """Extract and describe diagrams, always removing a unique run-specific temp dir."""
    print(f"Extracting images from {pdf_path}...")
    extractor = extractor or extract_images_from_pdf
    facts = []
    with tempfile.TemporaryDirectory(prefix="personal_kb_diagrams_", dir=temp_root) as output_dir:
        images = deduplicate_extracted_images(extractor(pdf_path, output_dir))
        print(f"Found {len(images)} images")
        for page_num, img_index, img_path in images:
            label = f"page {page_num}" + (f", image {img_index}" if img_index > 0 else " (full page)")
            print(f"\nProcessing {label}...")
            try:
                description, diagram_type = describe_diagram(
                    img_path, api_url=api_url, model=model, transport=transport
                )
                print(f"  Type: {diagram_type}")
                print(f"  Description: {description[:150]}...")
                facts.append({
                    "content": description, "topic": f"diagram: {diagram_type}",
                    "type": "diagram", "course": course_code, "lecture": lecture,
                    "source_type": "diagram", "diagram_type": diagram_type,
                    "page": page_num, "source_file": os.path.basename(pdf_path), "engine": model,
                })
            except Exception as e:
                print(f"  Error: {e}")
    return facts


def main():
    parser = argparse.ArgumentParser(description="Extract and describe diagrams from PDFs")
    parser.add_argument("pdf", help="Path to PDF file")
    parser.add_argument("--course", required=True, help="Course code")
    parser.add_argument("--lecture", type=int, required=True, help="Lecture number")
    parser.add_argument("--output", default=None, help="Output JSON file path")
    parser.add_argument("--api-url", default=QWEN_VISION_API_URL, help="DashScope OpenAI-compatible endpoint")
    parser.add_argument("--model", default=VISION_MODEL, help="Vision model")

    args = parser.parse_args()

    facts = process_pdf_diagrams(
        args.pdf,
        args.course,
        args.lecture,
        api_url=args.api_url,
        model=args.model,
    )

    print(f"\nTotal diagram facts: {len(facts)}")

    # Save
    if args.output:
        output_path = args.output
    else:
        base = os.path.splitext(os.path.basename(args.pdf))[0]
        output_path = os.path.join(os.path.dirname(args.pdf), f"{base}_diagrams.json")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(facts, f, indent=2, ensure_ascii=False)

    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
