from datasets import load_dataset
from datasets import DatasetDict

import argparse


def main(args):
    yourdatasetname = args.yourdatasetname
    creator = args.creator
    print(f"Dataset Name: {args.yourdatasetname}")
    print(f"Creator: {args.creator}")

    # Load the dataset
    dataset = load_dataset(f"{creator}/{yourdatasetname}")

    def format_function(example):
        return {
            'image': example['image'],  
            'solution': example['caption'],
            'original_question': "",
            'original_answer': "",
            'question' : "Whats on this image?",
            'language' : "dk",
            'source'   : f"{creator}/{yourdatasetname}"
        }

    # Print the original features to see the structure
    print("Original features:", dataset['train'].features)

    # Apply the formatting function to the dataset
    formatted_dataset = dataset.map(format_function)

    # Print the features of the formatted dataset to confirm changes
    print("Formatted features:", formatted_dataset['train'].features)

    # Specify the path where you want to save the dataset in Parquet format
    path_to_save = f"./{yourdatasetname}"

    # Push the dataset to your Hugging Face Hub repository
    formatted_dataset.push_to_hub(f"MykMaks/{yourdatasetname}")

    # Save the formatted dataset to disk in Parquet format
    formatted_dataset.set_format(type='arrow')
    formatted_dataset.save_to_disk(path_to_save)


if __name__ == "__main__":
    # Create the parser
    parser = argparse.ArgumentParser(description="Process some parameters.")

    # Add arguments
    parser.add_argument(
        "--yourdatasetname",
        type=str,
        default="nordjylland-news-image-captioning",
        help="The name of the dataset (default: nordjylland-news-image-captioning)"
    )
    parser.add_argument(
        "--creator",
        type=str,
        default="alexandrainst",
        help="Creator of the dataset (default: alexandrainst)"
    )

    # Parse arguments
    args = parser.parse_args()

    # Call the main function
    main(args)
