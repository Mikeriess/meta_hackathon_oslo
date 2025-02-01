import os
import random
import hashlib
import numpy as np
from PyPDF2 import PdfReader, PdfWriter
from pdf2image import convert_from_path
from PIL import Image, ImageEnhance, ImageFilter
from docling.document_converter import DocumentConverter
from tqdm import tqdm

def apply_distortions(image, rotate=0, noise=False, color=False, blur=False):
    """Applies optional distortions to the image."""
    if rotate:
        angle = random.uniform(-rotate, rotate)
        image = image.rotate(angle, resample=Image.BICUBIC, fillcolor=(255, 255, 255))
    
    if color:
        enhancer = ImageEnhance.Color(image)
        image = enhancer.enhance(random.uniform(0.8, 1.3))
    
    if noise:
        np_image = np.array(image)
        noise_array = np.random.randint(-30, 30, np_image.shape, dtype=np.int16)
        np_image = np.clip(np_image + noise_array, 0, 255).astype(np.uint8)
        image = Image.fromarray(np_image)
    
    if blur:
        image = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.8, 1.3)))
    
    return image

def split_pdf(input_pdf, output_folder):
    """Splits a multi-page PDF into single-page PDFs."""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    reader = PdfReader(input_pdf)
    base_hash = hashlib.md5(input_pdf.encode()).hexdigest()

    single_page_pdfs = []
    for i, page in enumerate(reader.pages):
        writer = PdfWriter()
        writer.add_page(page)

        output_pdf_path = os.path.join(output_folder, f"{base_hash}_{i+1}.pdf")
        with open(output_pdf_path, "wb") as output_pdf:
            writer.write(output_pdf)
    
        single_page_pdfs.append(output_pdf_path)

    return single_page_pdfs, base_hash

def convert_pdf_to_images(pdf_path, output_folder, rotate=0, noise=False, color=False, blur=False, hash=""):
    """Converts a single-page PDF to an image and extracts text to Markdown."""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    base_hash = hash
    page = int(pdf_path.split("_")[-1].replace(".pdf", ""))

    image_path = os.path.join(output_folder, f"{base_hash}_{page}.png")

    # Skip processing if the image already exists
    if os.path.exists(image_path):
        print(f"✅ Skipping {image_path} (already processed)")
        return

    images = convert_from_path(pdf_path)
    
    for image in images:
        if any([rotate, noise, color, blur]):
            image = apply_distortions(image, rotate, noise, color, blur)

        image.save(image_path, "PNG")
        print(f"📄 Saved Image: {image_path}")

def process_pdf_with_docling(pdf_path, output_folder):
    """Processes a PDF using Docling and saves the result as Markdown in the output folder."""
    docling_output_dir = os.path.join(output_folder, "markdown")
    os.makedirs(docling_output_dir, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(pdf_path))[0]
    docling_md_path = os.path.join(docling_output_dir, f"{base_name}_docling.md")

    # Skip processing if the markdown already exists
    if os.path.exists(docling_md_path):
        print(f"✅ Skipping {docling_md_path} (already processed)")
        return

    converter = DocumentConverter()
    result = converter.convert(pdf_path)
    markdown_output = result.document.export_to_markdown()

    with open(docling_md_path, "w", encoding="utf-8") as md_file:
        md_file.write(markdown_output)

    print(f"📝 Saved Docling Markdown: {docling_md_path}")

def make_dataset(source_pdf):
    output_dir = "output"
    tmp_dir = "tmp"
    rotate = 4
    noise = True
    color = True
    blur = True

    # Step 1: Split PDF into single-page PDFs
    print("\n🔹 Splitting PDF into single pages...")
    single_page_pdfs, hash = split_pdf(source_pdf, tmp_dir)

    # Step 2: Convert each single-page PDF into images & Markdown
    print("\n🔹 Converting single-page PDFs to images & Markdown...")
    for pdf in tqdm(single_page_pdfs, desc="Processing Images & Markdown"):
        convert_pdf_to_images(pdf, output_dir, rotate, noise, color, blur, hash)

    # Step 3: Process each single-page PDF with Docling
    print("\n🔹 Processing with Docling...")
    for pdf in tqdm(single_page_pdfs, desc="Running Docling Conversion"):
        process_pdf_with_docling(pdf, output_dir)
    
    # cleanup temporary directory
    print("\n🔹 Cleaning up temporary files...")
    for pdf in single_page_pdfs:
        os.remove(pdf)


    print("\n✅ All tasks completed!")


if __name__ == "__main__":
    # Get all pdf files in input folder
    input_folder = "input/updated"
    pdf_files = [f for f in os.listdir(input_folder) if f.endswith(".pdf")]
    # Process each PDF file

    for pdf_file in tqdm(pdf_files):
        source_pdf = os.path.join(input_folder, pdf_file)
        print(f"\n📄 Processing PDF: {source_pdf}")
        make_dataset(source_pdf)
        print("\n🎉 All PDFs processed")
