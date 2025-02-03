**NOTE:** This is the submission of team "MykMaks" at the Meta Llama Hackathon in Oslo 2025.

# 🇳🇴+🇩🇰 Open Source Vision Language Model assets
Building on the philosophy of open source with the Llama-models 🦙, this repo is an effort to support development of small VLM's in the Scandinavian languages. Aa we are only fluent in Norwegian and Danish, we have focused on these two languages. However, we encourgage the community to help build on our work and extend the coverage. 

💪 The current models and data focus on transcription and annotiation of documents in Norwegian and Danish, going beyond the limitations of OCR.

Code is seperated into [data preperation/processing](/data-preperation/) and [finetuning a Vision LLM on a single GPU](/llama_32_finetuning/).

We expect this line of work to help businesses, government institutions and citizens alike. Please see this chainlit documentation and video for how to run inference on the final models.

# Markdown transcription of Norwegian and Danish images
- 💽 Datasets for final fine-tune
  - 🇳🇴 [Norwegian dataset collection](https://huggingface.co/collections/MykMaks/datasets-nb-679f081d89be13de6a9fe71b): A collection of Norwegian datasets with a focus on complex diagrams and tables.
  - 🇩🇰 [Danish dataset collection](https://huggingface.co/collections/MykMaks/datasets-da-679f07b68e587e67bba71fdd): Data from newspapers and the Danish Ministry of Finance's reports with numbers, complex diagrams and tables.
- 💽 [Prompt examples used for finetuning](https://github.com/Mikeriess/meta_hackathon_oslo/blob/main/llama_32_finetuning/docker_vm/workspace/experiments.json) used with **SFTTrainer** and **UnslothVisionDataCollator**
  - **🇩🇰 MykMaks/fmsudgivelser**:
    - Beskriv venligst dette billede.
  - **🇩🇰 MykMaks/da-wit**
    - Beskriv hvad du ser i dette billede.
  - **🇳🇴 MykMaks/NorwegianDataset-compressed**
    - Vennligst beskriv hva du ser i dette bildet. 
  - **🇳🇴 MykMaks/NorwegianDataset-compressed-pt2**
    - "Produser markdown text fra dette dokumentet"
- 💾 Training code for finetune
  - Approach: We trained every epoch with a different prompt, stored the adapter as a checkpoint and continued to next prompt-dataset pair.
  - MM checkpoints: https://github.com/Mikeriess/llama33_resources/tree/MM-models
  - V-I checkpoints: https://github.com/Mikeriess/llama33_resources/tree/v-i-models
- 🤖 [Finetuned models from LORA-adapter checkpoints of Llama-3.2-11B-Vision-Instruct](https://huggingface.co/collections/MykMaks/models-679f08ab3ea3e21df62c87e8)
  - The model is iteratively trained over all datasets
    - The suffix of each file denotes the order of the checkpoint, along with the dataset that it was fine-tuned on
- 💸 Final full-precision merged models:
  - See collection: 🦙 https://huggingface.co/collections/MykMaks/models-679f08ab3ea3e21df62c87e8
    - <b>MykMaks/llama-3.2-11B-MM-20-MykMaks_da-wit-merged</b>
    - <b>MykMaks/llama-3.2-11B-V-I_39_MykMaks_NorwegianDataset-compressed-pt2-merged</b>

# Contributions/Assets
You can find all the assets (models, adapters, code and data) at:
[huggingface](https://huggingface.co/MykMaks)
