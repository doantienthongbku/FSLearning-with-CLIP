from pydoc import cli
import numpy as np
import torch
import os
from pkg_resources import packaging
import clip
import glob
from PIL import Image


def prepare_weight(model, support_dir: str, save_dir: str, device: str):
    # get text features embedding
    feature_embeddings = {}
    for label in os.listdir(support_dir):
        text_features = []
        for image_path in glob.glob(os.path.join(support_dir, label, "*.jpg")):
            image = Image.open(image_path).convert('RGB')
            image = preprocess(image).unsqueeze(0).to(device)
            
            with torch.no_grad():
                text_features.append(model.encode_image(image))
        
        feature_embeddings[label] = torch.mean(torch.stack(text_features), dim=0)
        # save to checkpoint
        torch.save(feature_embeddings[label], os.path.join(save_dir, f"{label}.pt"))

    
if __name__=='__main__':
    print(clip.available_models())
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, preprocess = clip.load("ViT-L/14@336px")
    model.to(device).eval()
    
    support_dir = "data/support"
    save_dir = "checkpoints"
    
    prepare_weight(model, support_dir, save_dir, device)