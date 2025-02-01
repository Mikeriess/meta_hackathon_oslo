import os
import argparse
from datasets import Dataset
from PIL import Image as PILImage
import markdown
from bs4 import BeautifulSoup

def parse_args():
    parser = argparse.ArgumentParser(description="Create and optionally upload a dataset from images and markdown files.")
    parser.add_argument("--data_dir_image", type=str, default='/home/theoo/meta_hackathon_oslo/convertpdf/output', help="Directory containing image files.")
    parser.add_argument("--data_dir_mds", type=str, default='/home/theoo/meta_hackathon_oslo/convertpdf/output/markdown', help="Directory containing markdown files.")
    parser.add_argument("--dataset_name", type=str, default="fmsudgivelser", help="Name of the dataset.")
    parser.add_argument("--push_to_hub", type=bool, default=True, help="Whether to push the dataset to Hugging Face Hub (default: True).")
    return parser.parse_args()

def read_image(image_path):
    """Read an image file and convert it to RGB."""
    return PILImage.open(image_path).convert('RGB')

def read_markdown(md_path):
    """Read a markdown file and convert it to plain text."""
    with open(md_path, 'r', encoding='utf-8') as file:
        md_content = file.read()
    html_content = markdown.markdown(md_content)
    soup = BeautifulSoup(html_content, "html.parser")
    return soup.get_text()

def main():
    args = parse_args()

    # List image and markdown files
    image_files = [f for f in os.listdir(args.data_dir_image) if f.endswith('.jpg') or f.endswith('.png')]
    md_files = [f for f in os.listdir(args.data_dir_mds) if f.endswith('.md')]

    # Prepare data for the dataset
    data = []
    for image_file in image_files:
        base_name = os.path.splitext(image_file)[0]
        md_file = f"{base_name}_docling.md"
        if md_file in md_files:
            image_path = os.path.join(args.data_dir_image, image_file)
            md_path = os.path.join(args.data_dir_mds, md_file)
            image = read_image(image_path)
            caption = read_markdown(md_path)
            data.append({
                'image': image,  
                'solution': caption,
                'original_question': "",
                'original_answer': "",
                'question': "Whats on this image?",
                'language': "dk",
                'source': f"fm/udgivelser"
            })

    print(f"length of dataset: ", len(data))

    # Convert list of dictionaries to a dictionary of lists
    data_dict = {key: [dic[key] for dic in data] for key in data[0]}
    dataset = Dataset.from_dict(data_dict)

    # Save the dataset to disk
    path_to_save = f"./{args.dataset_name}"
    dataset.save_to_disk(path_to_save)

    # Optionally, push the dataset to your Hugging Face Hub repository
    if args.push_to_hub:
        dataset.push_to_hub(f"MykMaks/{args.dataset_name}")

if __name__ == "__main__":
    main()
