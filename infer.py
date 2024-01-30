import os
import skimage
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import clip
import glob
from sklearn.metrics.pairwise import cosine_similarity

from collections import OrderedDict
import torch


def inference(model, preprocess, image_path, checkpoint_dir, device):
    """function to evaluate model performance

    Args:
        model: CLIP model
        preprocess: preprocess function of CLIP model
        image_dir: directory of images to be evaluated 
        checkpoint_dir: directory of checkpoints
        device: device to run the model 
    """
    
    # load checkpoints
    checkpoint_embedding_paths = glob.glob(os.path.join(checkpoint_dir, "*.pt"))
    embeddings = {}
    for checkpoint_embedding_path in checkpoint_embedding_paths:
        embeddings[os.path.basename(checkpoint_embedding_path).split('.')[0]] = torch.load(checkpoint_embedding_path)

    # inference
    image = Image.open(image_path).convert('RGB')
    image = preprocess(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        image_features = model.encode_image(image)
        
    # calculate distance using cosine similarity
    dist = {}
    for label, text_features in embeddings.items():
        dist[label] = cosine_similarity(image_features.cpu().numpy(), text_features.cpu().numpy())
    
    pred_label = max(dist, key=dist.get)
    print(f"Predicted label: {pred_label}")
    print(f"Confidence: {dist[pred_label][0][0]:.4f}")
    


if __name__=='__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, preprocess = clip.load("ViT-B/32")
    model.to(device).eval()
    image_path = "data/test/cat.119.jpg"
    checkpoint_dir = "checkpoints"

    inference(model, preprocess, image_path, checkpoint_dir, device)