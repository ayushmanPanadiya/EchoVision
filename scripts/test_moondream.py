
import torch
from PIL import Image
from transformers import AutoModelForCausalLM

MODEL_ID = "vikhyatk/moondream2"
REVISION = "2025-06-21"

print("GPU:", torch.cuda.get_device_name(0))
print("Loading Moondream 2...")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    revision=REVISION,
    trust_remote_code=True,
    torch_dtype=torch.float16,
    device_map={"": "cuda"},
)

print("Model loaded successfully!")

image_path = input(
    "Enter the full path to an image: "
).strip().strip('"')

image = Image.open(image_path).convert("RGB")
encoded_image = model.encode_image(image)

while True:
    question = input(
        "\nAsk a question (or type exit): "
    ).strip()

    if question.lower() == "exit":
        break

    result = model.query(encoded_image, question)
    print("\nMoondream:", result["answer"])
