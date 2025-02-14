from transformers import ViTModel, ViTFeatureExtractor, AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

class ImageModel:
    def __init__(self, use_resnet=False):
        self.use_resnet = use_resnet
        if use_resnet:
            self.model = models.resnet50(pretrained=True)
            self.model.eval()
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224")
            self.model = ViTModel.from_pretrained("google/vit-base-patch16-224")

    def preprocess(self, image):
        if self.use_resnet:
            image = self.transform(image).unsqueeze(0)  # Add batch dimension
        else:
            image = self.feature_extractor(images=image, return_tensors="pt")
        return image

    def extract_features(self, image):
        inputs = self.preprocess(image)
        with torch.no_grad():
            if self.use_resnet:
                outputs = self.model(inputs)
            else:
                outputs = self.model(**inputs)
        return outputs if self.use_resnet else outputs.last_hidden_state

class TeluguCaptionGenerator:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-te")
        self.model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-te")

    def generate_caption(self, english_caption):
        inputs = self.tokenizer(english_caption, return_tensors="pt", padding=True, truncation=True)
        outputs = self.model.generate(**inputs)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

def describe_image(image_path, context, use_resnet=False):
    image = Image.open(image_path).convert("RGB")
    image_model = ImageModel(use_resnet=use_resnet)

    # Extract features
    features = image_model.extract_features(image)

    # Placeholder caption (replace with actual model-based description)
    english_caption = f"This image is related to: {context}"

    # Translate English caption to Telugu
    caption_generator = TeluguCaptionGenerator()
    telugu_caption = caption_generator.generate_caption(english_caption)

    return telugu_caption

# Example usage
if __name__ == "__main__":
    image_path = "example.jpg"  # Replace with your image path
    context = "A beautiful sunset over the ocean."
    result = describe_image(image_path, context, use_resnet=False)
    print("Telugu Description:", result)
