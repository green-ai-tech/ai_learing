#!/usr/bin/env python3
"""Extract base64-encoded images from Jupyter notebooks and save as PNG files."""

import json
import base64
import os

OUTPUT_DIR = "/Users/logicye/Code/ai_learning/assets/images"
os.makedirs(OUTPUT_DIR, exist_ok=True)

NOTEBOOK_DIR = "/Users/logicye/Code/ai_learning/notebooks/01_machine_vision"

extractions = [
    {
        "notebook": "04_lenet5_handwritten_digit_recognition.ipynb",
        "output_file": "04_lenet5_mnist_sample.png",
        "description": "MNIST digit image from cell output (image/png)",
        "method": "cell_output",
        # Cell with execution_count 22, outputs[1] has the image/png
        "cell_index": None,  # We'll search by execution_count
        "execution_count": 22,
        "output_index": 1,
    },
    {
        "notebook": "03_image_processing_and_features.ipynb",
        "output_file": "03_sobel_edge_detection.png",
        "description": "Sobel edge detection image from cell output",
        "method": "cell_output",
        "execution_count": 6,
        "output_index": 1,
    },
    {
        "notebook": "05_pytorch_tensors_and_nn.ipynb",
        "output_file": "05_gradient_descent.png",
        "description": "Gradient descent convergence plot from cell output",
        "method": "cell_output",
        "execution_count": 28,
        "output_index": 0,
    },
    {
        "notebook": "02_video_background_replacement.ipynb",
        "output_file": "02_yolo_architecture.png",
        "description": "YOLO architecture/attachment image",
        "method": "attachment",
        "attachment_key": "c4721f91-03cb-40d6-8336-281537f3256c.png",
    },
    {
        "notebook": "06_model_training_finetuning_01_pretraining_transfer_learning.ipynb",
        "output_file": "06_coco_annotation.png",
        "description": "COCO dataset annotation format illustration (attachment)",
        "method": "attachment",
        "attachment_key": "2bc78d98-9c4d-48a2-9a46-bb4af6c633df.png",
    },
    {
        "notebook": "08_model_postprocessing_and_architecture.ipynb",
        "output_file": "08_pipeline_diagram.png",
        "description": "Pipeline working mechanism diagram (attachment)",
        "method": "attachment",
        "attachment_key": "a8c97f9d-3f8f-4b26-82be-e9ed7fc7f9d6.png",
    },
]


def extract_from_cell_output(nb_path, exec_count, output_index):
    """Extract image/png from a specific cell's output."""
    with open(nb_path, "r") as f:
        nb = json.load(f)

    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        if cell.get("execution_count") != exec_count:
            continue
        outputs = cell.get("outputs", [])
        if output_index >= len(outputs):
            print(f"  ERROR: output_index {output_index} out of range (has {len(outputs)} outputs)")
            return None
        output = outputs[output_index]
        data = output.get("data", {})
        if "image/png" not in data:
            print(f"  ERROR: no 'image/png' key in output data. Keys: {list(data.keys())}")
            return None
        return data["image/png"]

    print(f"  ERROR: no cell with execution_count={exec_count} found")
    return None


def extract_from_attachment(nb_path, attachment_key):
    """Extract image from cell attachments."""
    with open(nb_path, "r") as f:
        nb = json.load(f)

    for cell in nb.get("cells", []):
        attachments = cell.get("attachments", {})
        if attachment_key in attachments:
            img_data = attachments[attachment_key]
            # Could be "image/png" or other MIME types
            for mime_type in ["image/png", "image/jpeg", "image/svg+xml"]:
                if mime_type in img_data:
                    return img_data[mime_type]
            # Return first available
            first_key = next(iter(img_data), None)
            if first_key:
                return img_data[first_key]

    print(f"  ERROR: attachment '{attachment_key}' not found")
    return None


def main():
    for ext in extractions:
        nb_path = os.path.join(NOTEBOOK_DIR, ext["notebook"])
        out_path = os.path.join(OUTPUT_DIR, ext["output_file"])

        if not os.path.exists(nb_path):
            print(f"SKIP: {ext['notebook']} not found at {nb_path}")
            continue

        print(f"\nExtracting: {ext['output_file']}")
        print(f"  Notebook: {ext['notebook']}")
        print(f"  Description: {ext['description']}")

        base64_data = None
        if ext["method"] == "cell_output":
            base64_data = extract_from_cell_output(
                nb_path, ext["execution_count"], ext["output_index"]
            )
        elif ext["method"] == "attachment":
            base64_data = extract_from_attachment(nb_path, ext["attachment_key"])

        if base64_data is None:
            print(f"  FAILED: could not extract image data")
            continue

        try:
            image_bytes = base64.b64decode(base64_data)
            with open(out_path, "wb") as f:
                f.write(image_bytes)
            print(f"  SUCCESS: saved {len(image_bytes)} bytes to {out_path}")
        except Exception as e:
            print(f"  FAILED: {e}")


if __name__ == "__main__":
    main()
