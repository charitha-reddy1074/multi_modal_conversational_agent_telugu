from PIL import Image
import numpy as np
import torchvision.transforms as transforms

def preprocess_image(image_path):
    # Load the image
    image = Image.open(image_path).convert('RGB')
    
    # Define the transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to 224x224
        transforms.RandomHorizontalFlip(),  # Data augmentation
        transforms.ToTensor(),  # Convert to tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
    ])
    
    # Apply transformations
    image_tensor = transform(image)
    
    return image_tensor.numpy()  # Return as numpy array for further processing

def extract_embeddings(image_tensor, model):
    with torch.no_grad():
        embeddings = model(image_tensor.unsqueeze(0))  # Add batch dimension
    return embeddings.numpy()  # Return embeddings as numpy array