from torch import nn

class FusionModel(nn.Module):
    def __init__(self, image_model, audio_model, text_model):
        super(FusionModel, self).__init__()
        self.image_model = image_model
        self.audio_model = audio_model
        self.text_model = text_model
        self.fc = nn.Linear(image_model.output_dim + audio_model.output_dim + text_model.output_dim, 256)
        self.output_layer = nn.Linear(256, 1)  # Adjust output size as needed

    def forward(self, image_input, audio_input, text_input):
        image_features = self.image_model(image_input)
        audio_features = self.audio_model(audio_input)
        text_features = self.text_model(text_input)

        # Concatenate features from all modalities
        combined_features = torch.cat((image_features, audio_features, text_features), dim=1)
        fused_output = self.fc(combined_features)
        final_output = self.output_layer(fused_output)

        return final_output

    def get_image_model(self):
        return self.image_model

    def get_audio_model(self):
        return self.audio_model

    def get_text_model(self):
        return self.text_model