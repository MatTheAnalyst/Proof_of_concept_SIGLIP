import numpy as np
import pandas as pd

def complete(lst, padding_length):
    for i in range(padding_length - len(lst)):
        lst.append(f'add_{i}')
    return lst

def top_k_indices(arr, k):
    return np.argsort(arr[-k:])

def create_sample_uniform(data, categorie_name, each_cat_size):
  cat_distribution = data[categorie_name].value_counts()
  col_over_size = cat_distribution[cat_distribution>=each_cat_size].index
  cat_under_size = cat_distribution[cat_distribution<each_cat_size].index
  sample = pd.DataFrame()
  for cat in col_over_size:
    sample = pd.concat([sample, data[data[categorie_name]==cat].sample(n=each_cat_size, random_state=42)])
  sample = pd.concat([sample, data.loc[data[categorie_name].isin(cat_under_size)]])
  return sample