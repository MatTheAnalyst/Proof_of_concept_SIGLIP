import yaml
from PIL import Image
import faiss
import torch
from classes import Siglip
import pandas as pd
import numpy as np

with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

sample = pd.read_csv(config['data']['sample'])
images_path = sample['id'].to_list()
descriptions = sample['desc'].to_list()
del sample
siglip_model = Siglip(image_paths=images_path, descriptions=descriptions, model=config['models']['siglip'])
siglip_images_features = torch.load(config['tensors']['siglip_images'])
image_embeddings = siglip_images_features.numpy()
dimension = image_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(image_embeddings)

sample = pd.read_csv(config['data']['sample'])
images_path = sample['id'].to_list()
descriptions = sample['desc'].to_list()
del sample
siglip_model = Siglip(image_paths=images_path, descriptions=descriptions, model=config['models']['siglip'])

try:
    inputs = siglip_model.processor(text=["Veste noire pour homme"], images=Image.open(siglip_model.image_paths[0]).convert("RGB"), padding="max_length", return_tensors="pt")
    print("Inputs processed successfully.")
except Exception as e:
    print(f"Erreur lors du traitement des inputs: {e}")
