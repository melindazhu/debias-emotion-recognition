import os
import torch
import numpy as np
import pandas as pd
from PIL import Image
from facenet_pytorch import InceptionResnetV1
import torchvision.transforms as transforms

# === File Paths (Generic) ===
raf_path = "./data/RAF_DB/" 
train_csv = "./data/RAF_DB/train_labels.csv"
test_csv = "./data/RAF_DB/test_labels.csv"

# === Load Pretrained FaceNet Model ===
# Load the FaceNet model pretrained on the VGGFace2 dataset for face embeddings
facenet = InceptionResnetV1(pretrained='vggface2').eval()  # Set to evaluation mode

# === Image Preprocessing Function ===
def preprocess_face(image_path):
    """
    Preprocesses the image to be compatible with FaceNet for embedding extraction.
    - Converts the image to RGB.
    - Resizes the image to 160x160 pixels (required input size for FaceNet).
    - Normalizes the pixel values for compatibility with the model.
    """    
    img = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    return transform(img).unsqueeze(0) # Add a batch dimension and return the tensor

# === Extract Face Embeddings ===
def get_embedding(model, image_path):
    """
    Extracts the face embedding from an image using FaceNet.
    - Preprocesses the image.
    - Passes the image through the FaceNet model to generate the embedding.
    - Returns the embedding as a numpy array.
    """    
    try:
        # Preprocess the image
        img_tensor = preprocess_face(image_path)
        with torch.no_grad(): # Disable gradient calculation for inference
            embedding = model(img_tensor).cpu().numpy()
        return embedding
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

# === Load RAF-DB Labels from CSV ===
# Load label information for training and testing from CSV files
train_df = pd.read_csv(train_csv)
test_df = pd.read_csv(test_csv)
raf_df = pd.concat([train_df, test_df], ignore_index=True)

# Create a dictionary mapping filenames to labels (assuming labels are in 'label' column)
raf_labels = dict(zip(raf_df['image'], raf_df['label'])) # Combine both training and test label DataFrames into a single DataFrame

# === Process RAF-DB Dataset ===
# Process each image in the RAF-DB dataset to generate embeddings and store them
raf_embeddings = {}
for split in ["train", "test"]:  # Navigate into train/test folders
    split_path = os.path.join(raf_path, split)

    for emotion_label in os.listdir(split_path):  # Iterate through emotion folders (1,2,3..7)
        emotion_folder = os.path.join(split_path, emotion_label)

        if not os.path.isdir(emotion_folder):  # Skip if not a directory
            continue

        for img_file in os.listdir(emotion_folder):  # Process images
            img_path = os.path.join(emotion_folder, img_file)
            embedding = get_embedding(facenet, img_path)
            if embedding is not None:
                raf_embeddings[img_file] = embedding
                raf_labels[img_file] = raf_labels.get(img_file, "Unknown")  # Assign emotion label if found

# === Save Embeddings & Labels ===
np.save("./outputs/raf_embeddings.npy", raf_embeddings)
np.save("./outputs/raf_labels.npy", raf_labels)

print("✅ Processing complete! Data saved in output directory.")
