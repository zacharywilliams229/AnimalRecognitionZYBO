import os
import json
import torch
import numpy as np

# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "models/animalcnn_val_acc_model.pth"
EXPORT_DIR = "exported_models"

# Input tensor dimensions
INPUT_SHAPE = (1, 3, 128, 128)

# Quantization scale
# Example:
# float_value * SCALE -> int8
SCALE = 127.0

# Save intermediate layer outputs for debugging
SAVE_REFERENCE_OUTPUTS = True

# ============================================================
# IMPORT YOUR MODEL
# ============================================================

from model import AnimalCNN

# ============================================================
# CREATE EXPORT DIRECTORY
# ============================================================

os.makedirs(EXPORT_DIR, exist_ok=True)

# ============================================================
# LOAD MODEL
# ============================================================

model = AnimalCNN()
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()

print("Model loaded successfully.")

# ============================================================
# HELPER FUNCTIONS
# ============================================================


def quantize_tensor(tensor, scale=SCALE):
    """
    Convert float tensor to int8.
    """

    tensor = tensor.detach().cpu().numpy()

    quantized = np.round(tensor * scale)

    quantized = np.clip(quantized, -128, 127)

    return quantized.astype(np.int8)



def export_binary(array, filename):
    """
    Export array to raw binary.
    """

    path = os.path.join(EXPORT_DIR, filename)

    array.tofile(path)

    print(f"Saved binary: {path}")



def export_header(array, variable_name, filename):
    """
    Export array as C header.
    """

    flat = array.flatten()

    path = os.path.join(EXPORT_DIR, filename)

    with open(path, "w") as f:
        f.write("#ifndef GENERATED_WEIGHTS_H\n")
        f.write("#define GENERATED_WEIGHTS_H\n\n")

        f.write("#include <stdint.h>\n\n")

        f.write(f"const int8_t {variable_name}[] = {{\n")

        for i, value in enumerate(flat):
            f.write(str(int(value)))

            if i != len(flat) - 1:
                f.write(",")

            if (i + 1) % 16 == 0:
                f.write("\n")

        f.write("\n};\n\n")

        f.write(f"const int {variable_name}_len = {len(flat)};\n\n")

        f.write("#endif\n")

    print(f"Saved header: {path}")


# ============================================================
# EXPORT MODEL PARAMETERS
# ============================================================

metadata = {}

for name, param in model.named_parameters():

    print(f"Exporting: {name}")

    quantized = quantize_tensor(param)

    safe_name = name.replace('.', '_')

    # Save binary
    export_binary(quantized, f"{safe_name}.bin")

    # Save header
    export_header(
        quantized,
        safe_name,
        f"{safe_name}.h"
    )

    # Save metadata
    metadata[safe_name] = {
        "shape": list(param.shape),
        "dtype": "int8",
        "scale": SCALE
    }

# ============================================================
# SAVE METADATA
# ============================================================

metadata_path = os.path.join(EXPORT_DIR, "metadata.json")

with open(metadata_path, "w") as f:
    json.dump(metadata, f, indent=4)

print(f"Saved metadata: {metadata_path}")

# ============================================================
# SAVE REFERENCE INPUT/OUTPUT
# ============================================================

if SAVE_REFERENCE_OUTPUTS:

    print("Generating reference tensors...")

    dummy_input = torch.randn(INPUT_SHAPE)

    # Save float input
    input_np = dummy_input.detach().cpu().numpy()

    np.save(os.path.join(EXPORT_DIR, "reference_input.npy"), input_np)

    # Save quantized input
    quantized_input = quantize_tensor(dummy_input)

    export_binary(quantized_input, "reference_input.bin")

    # ========================================================
    # Capture intermediate outputs
    # ========================================================

    intermediate_outputs = {}

    def save_output(name):
        def hook(module, input, output):
            intermediate_outputs[name] = output.detach().cpu().numpy()
        return hook

    hooks = []

    for name, module in model.named_modules():

        # Skip container modules
        if len(list(module.children())) == 0:
            hooks.append(module.register_forward_hook(save_output(name)))

    with torch.no_grad():
        final_output = model(dummy_input)

    # Remove hooks
    for hook in hooks:
        hook.remove()

    # Save intermediate outputs
    for layer_name, output in intermediate_outputs.items():

        safe_name = layer_name.replace('.', '_')

        np.save(
            os.path.join(EXPORT_DIR, f"{safe_name}_output.npy"),
            output
        )

        quantized_output = quantize_tensor(torch.tensor(output))

        export_binary(
            quantized_output,
            f"{safe_name}_output.bin"
        )

    # Save final output
    final_output_np = final_output.detach().cpu().numpy()

    np.save(
        os.path.join(EXPORT_DIR, "final_output.npy"),
        final_output_np
    )

    final_output_quantized = quantize_tensor(final_output)

    export_binary(final_output_quantized, "final_output.bin")

    print("Reference outputs generated.")

# ============================================================
# SUMMARY
# ============================================================

print("\n================================================")
print("EXPORT COMPLETE")
print("================================================")
print(f"Export directory: {EXPORT_DIR}")
print("")
print("Generated files:")
print("- Quantized weight binaries")
print("- FPGA-friendly C headers")
print("- Metadata JSON")
print("- Reference tensors for debugging")