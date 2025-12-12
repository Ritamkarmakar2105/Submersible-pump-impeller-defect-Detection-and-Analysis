from pathlib import Path
import ollama

# =============== CONFIG ===============

# Change this if `ollama list` shows just "llava"
MODEL_NAME = "llava:7b"

# Path to the image you want to test
IMAGE_PATH = Path(
    r"D:\submersible pump impeller defect Detection analysis\runs\detect\predict14\cast_def_0_8245.jpg"
)

# =============== LOGIC ================

def analyze_image_with_llava(image_path: Path, prompt: str) -> str:
    """
    Send an image + text prompt to a local LLaVA model via Ollama Python library.
    """
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found at: {image_path}")

    # For vision models, Ollama Python accepts image paths directly in `images`
    response = ollama.chat(
        model=MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": prompt,
                "images": [str(image_path)],  # pass path as string
            }
        ],
    )

    # Response structure: {'model': ..., 'message': {'role': 'assistant', 'content': '...'}, ...}
    return response["message"]["content"]


def main():
    prompt = (
        "You are an industrial inspector for submersible pump impellers. "
        "Look carefully at this image and describe any visible defects, "
        "their type (scratch, crack, porosity, etc.), the severity, and whether "
        "the part should be accepted, reworked, or rejected. Be concise."
    )

    print("\n🧠 Vision-Language Analysis Running...\n")
    answer = analyze_image_with_llava(IMAGE_PATH, prompt)
    print("📌 LLaVA Response:\n")
    print(answer)


if __name__ == "__main__":
    main()
