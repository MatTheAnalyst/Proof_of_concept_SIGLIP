import yaml
import streamlit as st
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

st.title('Recherche d\'image basée sur le texte')

# Champ de saisie pour le prompt textuel
user_input = st.text_input("Entrez votre description:")

if user_input:
    # Générez l'embedding et recherchez les images similaires
    embedding = siglip_model.text_embedding(user_input)
    print('embedding success')
    embedding = embedding / np.linalg.norm(embedding)
    D, I = index.search(embedding.reshape(1, -1), 3)
    
    closest_images = [images_path[i] for i in I[0]]
    
    # Affichage des images
    for img_path in closest_images:
        image = Image.open(img_path)
        st.image(image, caption=img_path, use_column_width=True)