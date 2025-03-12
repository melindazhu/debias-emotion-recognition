import os
import torch
import numpy as np
import pandas as pd
from PIL import Image
from facenet_pytorch import InceptionResnetV1
import torchvision.transforms as transforms

raf_path = "./data/RAF_DB/" 
train_csv = "./data/RAF_DB/train_labels.csv"
test_csv = "./data/RAF_DB/test_labels.csv"

facenet = InceptionResnetV1(pretrained='vggface2').eval()  

def preprocess_face(image_path):
    img = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    return transform(img).unsqueeze(0) 
    
def get_embedding(model, image_path):
    try:
        img_tensor = preprocess_face(image_path)
        with torch.no_grad(): 
            embedding = model(img_tensor).cpu().numpy()
        return embedding
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

train_df = pd.read_csv(train_csv)
test_df = pd.read_csv(test_csv)
raf_df = pd.concat([train_df, test_df], ignore_index=True)

raf_labels = dict(zip(raf_df['image'], raf_df['label'])) 

raf_embeddings = {}
for split in ["train", "test"]:  
    split_path = os.path.join(raf_path, split)

    for emotion_label in os.listdir(split_path):  
        emotion_folder = os.path.join(split_path, emotion_label)

        if not os.path.isdir(emotion_folder):  
            continue

        for img_file in os.listdir(emotion_folder):  
            img_path = os.path.join(emotion_folder, img_file)
            embedding = get_embedding(facenet, img_path)
            if embedding is not None:
                raf_embeddings[img_file] = embedding
                raf_labels[img_file] = raf_labels.get(img_file, "Unknown")  
                
np.save("./outputs/raf_embeddings.npy", raf_embeddings)
np.save("./outputs/raf_labels.npy", raf_labels)

print("✅ Processing complete! Data saved in output directory.")
