import os
from PIL import Image

# Define input and output folders
input_folder = "convertpdf/output"
output_folder = "convertpdf/compressed_output"

# Ensure the output folder exists
os.makedirs(output_folder, exist_ok=True)

# Compression quality (lower value means higher compression)
QUALITY = 70

def compress_images(input_folder, output_folder, quality=QUALITY):
    for filename in os.listdir(input_folder):
        if filename.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            try:
                with Image.open(input_path) as img:
                    img = img.convert("RGB")  # Ensure compatibility with JPEG format
                    img.save(output_path, "JPEG", quality=quality)
                    print(f"Compressed: {filename} -> {output_path}")
            except Exception as e:
                print(f"Error processing {filename}: {e}")

if __name__ == "__main__":
    compress_images(input_folder, output_folder)
    print("Compression complete.")