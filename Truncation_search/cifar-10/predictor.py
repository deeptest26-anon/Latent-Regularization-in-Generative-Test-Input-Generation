import torch
import torchvision.transforms as transforms
import timm
import torch.nn as nn
import numpy as np
from config import MODEL, num_classes, DEVICE
from tensorflow import keras 

class Predictor:
    # Load the pre-trained model on the correct device
    device = torch.device(DEVICE)

    model = timm.create_model("vit_base_patch16_384", pretrained=False)
    model.head = nn.Linear(model.head.in_features, num_classes)

    ckpt_path = MODEL
    checkpoint = torch.load(ckpt_path, map_location=device)

    @staticmethod
    def remove_module_prefix(state_dict):
        return {
            k[len('module.'):] if k.startswith('module.') else k: v
            for k, v in state_dict.items()
        }

    state_dict = remove_module_prefix.__func__(checkpoint['model'])
    model.load_state_dict(state_dict)

    model.to(device)

    @staticmethod
    def get_transform():
        return transforms.Compose([
            transforms.Resize(384),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465),
                (0.2023, 0.1994, 0.2010)
            ),
        ])

    @staticmethod
    def predict_datapoint(image, label):
        """
        Args:
            image (PIL Image): The image to classify
            label (int): The ground-truth label index
        Returns:
            accepted (bool): True if predicted == label, otherwise False
            confidence_expclass (float): The confidence of the correct (expected) class
            predictions (np.ndarray): Softmax probabilities for all classes
        """
        device = Predictor.device
        model = Predictor.model
        model.eval()

        transform = Predictor.get_transform()
        input_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(input_tensor)
            _, prediction1 = output.max(1)
            predictions = torch.nn.functional.softmax(output[0], dim=0).cpu()
            confidence_expclass = predictions[label].item()

        return (prediction1[0].item() == label), confidence_expclass, predictions.numpy()
