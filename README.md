# 🇳🇴+🇩🇰 Open Source Vision Language Model assets
Building on the philosophy of open source with the Llama-models 🦙, this repo is an effort to support development of small VLM's in the Scandinavian languages. Aa we are only fluent in Norwegian and Danish, we have focused on these two languages. However, we encourgage the community (🇫🇮🇸🇪🇫🇴🇮🇸🇬🇱Sami) to help build on our work and extend the coverage. 

The current models and data focus on transcription and annotiation of documents in Norwegian and Danish, going beyond the limitations of OCR.

We expect this line of work to help businesses, government institutions and citizens alike. Please se <repo> for how to run inference on the final models.

# In these collections you will find:
- 💽 Datasets for final fine-tune 
  - 🇳🇴 [NorwegianDataset Collection](https://huggingface.co/collections/MykMaks/datasets-nb-679f081d89be13de6a9fe71b): A collection of datasets from the national library of Norway with a focus on complex diagrams and tables.
  - 🇩🇰 [FMs Udgivelser](https://huggingface.co/collections/MykMaks/datasets-da-679f07b68e587e67bba71fdd): The Danish Ministry of Finance's reports with numbers, complex diagrams and tables.
- [Prompts for finetune](https://github.com/Mikeriess/meta_hackathon_oslo/blob/main/llama_32_finetuning/docker_vm/workspace/experiments.json) used with **SFTTrainer** and **UnslothVisionDataCollator**
  - **MykMaks/fmsudgivelser**: Beskriv venligst dette billede.
  - **MykMaks/NorwegianDataset-compressed**
    - Vennligst beskriv hva du ser i dette bildet.
    - Transkriber dette dokumentet
    - Produser markdown text fra dette dokumentet
  - **MykMaks/da-wit**
    - Beskriv hvad du ser i dette billede.
  - **MykMaks/NorwegianDataset-compressed-pt2**
    - "Transkriber dette dokumentet"
    - "Produser markdown text fra dette dokumentet"
    - "Gjør en transkripsjon av bildet"
    - "Hva står det her?",
    - "Gi meg et dokument med teksten i dette bildet",
    - "Skriv ned så nøyaktig du kan hva som står her",
    - "OCR dette bildet",
    - "Utfør OCR og vis dokumentet som markdown",
    - "Kan du lese og vise meg dette dokumentet",
    - Hva står i bildet",
    - "Gi meg teksten her",
    - "Gjennskap bildet i markdown",
    - "Les nøye igjennom og transkriber dokumentet",
    - "Hei, kan du fortelle meg hva som står her.",
    - "Gjør bildet om til tekst.",
    - "Transkriber",
    - "Gjør om til tekst",
    - "Les dette og gjennskap som markdown"
- 💾 Training code for finetune
  - MM checkpoints: https://github.com/Mikeriess/llama33_resources/tree/MM-models
  - V-I checkpoints: https://github.com/Mikeriess/llama33_resources/tree/v-i-models
- 🤖 [Finetuned models from LORA-adapter checkpoints of Llama-3.2-11B-Vision-Instruct](https://huggingface.co/collections/MykMaks/models-679f08ab3ea3e21df62c87e8)
  - The model is iteratively trained over all datasets
    - The suffix of each file denotes the order of the checkpoint, along with the dataset that it was fine-tuned on
- 💸 Final merged model:
  - [Llama-3.2-11B-Vision-Instruct-MykMaks]((https://huggingface.co/MykMaks/llama-3.2-11B-MM-20-MykMaks_da-wit-merged))

See more MykMaks hackathon data and contribute to the open-source community at [huggingface](https://huggingface.co/MykMaks)
