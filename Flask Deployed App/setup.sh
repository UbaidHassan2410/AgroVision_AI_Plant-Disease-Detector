#!/bin/bash

# Download the pre-trained model from Google Drive
# Hugging Face Spaces build step - runs when the Space is built
echo "Downloading pre-trained model..."
wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1NmqJ7Bm-c1X-YqMn2vTEesrhnFwA3Hf5' -O plant_disease_model_1_latest.pt 2>&1 || \
curl -L -o plant_disease_model_1_latest.pt 'https://docs.google.com/uc?export=download&id=1NmqJ7Bm-c1X-YqMn2vTEesrhnFwA3Hf5' 2>&1

echo "Model download completed."
