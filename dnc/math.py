import torch

"""
These functions are DNC specific and are not general math utilities.
"""

def cosine_similarity(u, v):
  return (u * v).sum(2) / (norm(u, 2) * norm(v, 2))

def norm(u):
  return torch.sqrt(u ** 2.sum(2))