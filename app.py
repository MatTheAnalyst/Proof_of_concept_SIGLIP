import yaml
import streamlit as st
from PIL import Image
import torch
from classes import Siglip, Clip
import pandas as pd
import numpy as np

with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

sample = pd.read_csv(config['data']['sample'])
image_paths = sample['id'].to_list()
descriptions = sample['desc'].to_list()
del sample
siglip_model = Siglip(image_paths=image_paths, descriptions=descriptions, model=config['models']['siglip'])
clip_model = Clip(image_paths = image_paths, descriptions=descriptions, model=config['models']['clip'])
st.title('Recherche d\'image basée sur votre demande  textuelle')

# Champ de saisie pour le prompt textuel
user_input = st.text_input("Entrez votre description:")

if user_input:
    cols_siglip, cols_clip = st.columns(2)
    with cols_siglip:
        st.subheader("SigLIP results:")
        siglip_closest_images, siglip_D, siglip_I = siglip_model.image_recommandation(user_input, config['tensors']['siglip_images'])
        siglip_cols = st.columns(3)
        siglip_distances = siglip_D[0]
        for col, img_path, distance in zip(siglip_cols, siglip_closest_images, siglip_distances):
            image = Image.open(img_path)
            col.image(image, use_column_width=True)
            col.write(f"Similarité: {abs(1-distance):.2f}")
    with cols_clip:
        st.subheader("CLIP results:")
        clip_closest_images, clip_D, clip_I = clip_model.image_recommandation(user_input, config['tensors']['clip_images'])
        clip_cols = st.columns(3)
        clip_distances = clip_D[0]
        for col, img_path, distance in zip(clip_cols, clip_closest_images, clip_distances):
            image = Image.open(img_path)
            col.image(image, use_column_width=True)
            col.write(f"Similarité: {abs(1-distance):.2f}")
