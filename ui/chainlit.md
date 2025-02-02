# meta_hackathon_oslo

# 🇳🇴🇩🇰 Open Source Vision Language Model assets
Building on the philosophy of open source with the Llama-models 🦙, this repo is an effort to support development of small VLM's in the Scandinavian languages. Aa we are only fluent in Norwegian and Danish, we have focused on these two languages. However, we encourgage the community (🇫🇮🇸🇪🇫🇴🇮🇸🇬🇱Sami) to help build on our work and extend the coverage. 

The current models and data focus on transcription and annotiation of documents in Norwegian and Danish, going beyond the limitations of OCR.

We expect this line of work to help businesses, government institutions and citizens alike. Please se <repo> for how to run inference on the final models.

# In these collections you will find:
- 💽 Datasets for fine-tuning VLM
  - 🇳🇴 MykMaks/NorwegianDataset-compressed
  - 🇩🇰 MykMaks/fmsudgivelser
- 💾 Training code
  - MM checkpoints: https://github.com/mikeriess/llama33_resources/
  - V-I checkpoints: https://github.com/mikeriess/llama33_resources/
- 🤖 Model LORA-adapter checkpoints for Llama-3.2-11B-Vision-Instruct
  - The model is iteratively trained over all datasets:
    - The suffix of each file denotes the order of the checkpoint, along with the dataset that it was fine-tuned on
- 💸 Final merged model:
  - <b>Llama-3.2-11B-Vision-Instruct-MykMaks</b>