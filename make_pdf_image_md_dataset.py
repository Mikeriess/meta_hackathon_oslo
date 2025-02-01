import os
from datasets import Dataset, Image
from PIL import Image as PILImage
import markdown
from bs4 import BeautifulSoup


# Path to your images and markdown files
data_dir_image = '/home/theoo/meta_hackathon_oslo/convertpdf/output'
data_dir_mds = '/home/theoo/meta_hackathon_oslo/convertpdf/output/markdown'
yourdatasetname = "fmsudgivelser"

image_files = [f for f in os.listdir(data_dir_image) if f.endswith('.jpg') or f.endswith('.png')]
md_files = [f for f in os.listdir(data_dir_mds) if f.endswith('.md')]

# Function to read an image file
def read_image(image_path):
    return PILImage.open(image_path).convert('RGB')

def read_markdown(md_path):
    with open(md_path, 'r', encoding='utf-8') as file:
        md_content = file.read()
    html_content = markdown.markdown(md_content)
    soup = BeautifulSoup(html_content, "html.parser")
    return soup.get_text()

# Prepare data for the dataset
data = []
for image_file in image_files:
    base_name = os.path.splitext(image_file)[0]
    md_file = f"{base_name}_docling.md"
    if md_file in md_files:
        image_path = os.path.join(data_dir_image, image_file)
        md_path = os.path.join(data_dir_mds, md_file)
        image = read_image(image_path)  # This reads the image into memory
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
# Create a dataset
dataset = Dataset.from_dict(data_dict)

# Save the dataset to disk
path_to_save = f"./{yourdatasetname}"
dataset.save_to_disk(path_to_save)

# Optionally, push the dataset to your Hugging Face Hub repository
dataset.push_to_hub(f"MykMaks/{yourdatasetname}")
