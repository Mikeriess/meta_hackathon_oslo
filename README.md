# 🇳🇴+🇩🇰 Open Source Vision Language Model assets
Building on the philosophy of open source with the Llama-models 🦙, this repo is an effort to support development of small VLM's in the Scandinavian languages. Aa we are only fluent in Norwegian and Danish, we have focused on these two languages. However, we encourgage the community (🇫🇮🇸🇪🇫🇴🇮🇸🇬🇱Sami) to help build on our work and extend the coverage. 

The current models and data focus on transcription and annotiation of documents in Norwegian and Danish, going beyond the limitations of OCR.

We expect this line of work to help businesses, government institutions and citizens alike. Please se <repo> for how to run inference on the final models.

# In these collections you will find:
- 💽 Datasets for final fine-tune 
  - 🇳🇴 [MykMaks/NorwegianDataset-compressed](https://huggingface.co/datasets/MykMaks/NorwegianDataset-compressed)
  - 🇩🇰 [MykMaks/fmsudgivelser](https://huggingface.co/datasets/MykMaks/fmsudgivelser)
- Prompts for finetune
  - **MykMaks/fmsudgivelser**: Beskriv venligst dette billede.
  - **MykMaks/NorwegianDataset-compressed**
    - Vennligst beskriv hva du ser i dette bildet.
    - Transkriber dette dokumentet
    - Produser markdown text fra dette dokumentet
  - **MykMaks/da-wit**
    - Beskriv hvad du ser i dette billede.
  - **MykMaks/NorwegianDataset-compressed-pt2**
    - Transkriber dette dokumentet
    - Produser markdown text fra dette dokumentet
    - Gjør en transkripsjon av bildet
    - (continue listing all prompts...)
- 💾 Training code for finetune
  - MM checkpoints: https://github.com/Mikeriess/llama33_resources/tree/MM-models
  - V-I checkpoints: https://github.com/Mikeriess/llama33_resources/tree/v-i-models
- 🤖 Model LORA-adapter checkpoints for Llama-3.2-11B-Vision-Instruct
  - The model is iteratively trained over all datasets:
    - The suffix of each file denotes the order of the checkpoint, along with the dataset that it was fine-tuned on
- 💸 Final merged model:
  - [Llama-3.2-11B-Vision-Instruct-MykMaks](https://huggingface.co/MykMaks)

  See more at https://huggingface.co/MykMaks
