import pandas as pd
import numpy as np
import clip
import os
import torch
from tqdm import tqdm
from PIL import Image
from transformers import AutoProcessor, AutoModel

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
device = "cuda" if torch.cuda.is_available() else "cpu"


class Dataloader():
  def __init__(self, data_dir):
    self.data_dir = data_dir

  def load_data(self):
    data = pd.read_csv(self.data_dir)
    return data

  def get_images_path(self, sample_size=None):
    if sample_size:
      images_path = self.load_data()['id'].sample(n=sample_size, random_state=42).to_list()
    else:
      images_path = self.load_data()['id'].to_list()
    return images_path

  def get_descriptions(self, sample_size=None):
    if sample_size:
      descriptions = self.load_data()['desc'].sample(n=sample_size, random_state=42).to_list()
    else:
      descriptions = self.load_data()['desc'].to_list()
    return descriptions

class Clip():
    def __init__(self, image_paths, descriptions, model):
      self.image_paths = image_paths
      self.descriptions = descriptions
      self.model, self.processor = clip.load(model,device=device)

    def info(self):
      input_resolution = self.model.visual.input_resolution
      context_length = self.model.context_length
      vocab_size = self.model.vocab_size
      print("Models available:", f"{clip.available_models()}")
      print("Model parameters:", f"{np.sum([int(np.prod(p.shape)) for p in self.model.parameters()]):,}")
      print("Input resolution:", input_resolution)
      print("Context length:", context_length)
      print("Vocab size:", vocab_size)
      return None

    def get_text_features(self, batch_size):
      text_features = torch.empty(size=(len(self.descriptions), 512))
      for batch in tqdm(range(0, len(self.descriptions), batch_size)):
        description_batch = self.descriptions[batch:batch+batch_size]
        text_tokens = clip.tokenize(description_batch, truncate=True).to(device)
        with torch.no_grad():
          text_features[batch:batch+batch_size] = self.model.encode_text(text_tokens).float()
      return text_features

    def get_image_features(self, batch_size):
      image_features = torch.empty(size=(len(self.image_paths), 512))
      for batch in tqdm(range(0, len(self.image_paths), batch_size)):
        images_preprocessed = list()
        for idx, image in enumerate(self.image_paths[batch:batch+batch_size]):
            image = Image.open(image).convert("RGB")
            images_preprocessed.append(self.processor(image))
        inputs = torch.tensor(np.stack(images_preprocessed)).to(device)
        with torch.no_grad():
          image_features[batch:batch+batch_size] = self.model.encode_image(inputs).float()
      return image_features

class Siglip():
    def __init__(self, image_paths, descriptions, model):
      self.image_paths = image_paths
      self.descriptions = descriptions
      self.model = AutoModel.from_pretrained(model)
      self.processor = AutoProcessor.from_pretrained(model)

    def get_text_features(self, batch_size):
      image = Image.open(self.image_paths[0]).convert("RGB")
      text_features = torch.empty(size=(len(self.image_paths), 768))
      for batch in tqdm(range(0, len(self.descriptions), batch_size)):
        texts = self.descriptions[batch:batch+batch_size]
        inputs = self.processor(text=texts, images=image, padding="max_length", return_tensors="pt")
        with torch.no_grad():
            pixel_values = inputs.pixel_values.to(device)
            input_ids = inputs.input_ids.to(device)
            outputs = self.model(**inputs)
            del pixel_values, input_ids
        text_features[batch:batch+batch_size] = outputs.text_embeds
      return text_features

    def get_image_features(self):
      image_features = torch.empty(size=(len(self.image_paths), 768))
      text = self.descriptions[0]
      for idx, image in tqdm(enumerate(self.image_paths)):
        image = Image.open(image).convert("RGB")
        inputs = self.processor(text=text, images=image, padding="max_length", return_tensors="pt")
        with torch.no_grad():
            print("Image features computing")
            pixel_values = inputs.pixel_values.to(device)
            print("Text features computing")
            input_ids = inputs.input_ids.to(device)
            outputs = self.model(**inputs)
            del pixel_values, input_ids
        image_features[idx] = outputs.image_embeds.squeeze(0)
      return image_features

    def image_text_similarity(self, image_path, texts:list[str]):
      image = Image.open(image_path).convert("RGB")
      inputs = self.processor(text=texts, images=image, padding="max_length", return_tensors="pt")
      with torch.no_grad():
        outputs = self.model(**inputs)
      return outputs.text_embeds, outputs.image_embeds

    def text_embedding(self, text:str):
      image = Image.open(self.image_paths[0]).convert("RGB")
      text = [text]
      try:
        inputs = self.processor(text=text, images=image, padding="max_length", return_tensors="pt")
        print("Inputs processed successfully.")
      except Exception as e:
        print(f"Erreur lors du traitement des inputs: {e}")
        return None
      with torch.no_grad():
          pixel_values = inputs.pixel_values.to(device)
          input_ids = inputs.input_ids.to(device)
          outputs = self.model(**inputs)
      description_norm = outputs.text_embeds / outputs.text_embeds.norm(dim=1, keepdim=True)
      return description_norm

class ModelEvaluation():
  def __init__(self):
    pass

  def get_similarity(self,image_features_path, text_features_path):
    text_features = torch.load(text_features_path)
    image_features = torch.load(image_features_path)

    description_norm = text_features / text_features.norm(dim=1, keepdim=True)
    image_norm = image_features / image_features.norm(dim=1, keepdim=True)

    cosinus_similarity = torch.mm(description_norm, image_norm.transpose(0, 1))
    return cosinus_similarity

  def top_k_accuracy(self, cosine_similarity_matrix, k=1):
      top_k_predictions = torch.topk(cosine_similarity_matrix, k, dim=1).indices

      true_indices = torch.arange(cosine_similarity_matrix.size(0)).unsqueeze(1).expand_as(top_k_predictions)
      correct_predictions = top_k_predictions == true_indices
      top_k_accuracy = correct_predictions.any(dim=1).float().mean().item()
      print(f"Top-{k} Accuracy: {top_k_accuracy * 100:.2f}%")
      return top_k_accuracy