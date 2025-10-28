import torch
import torchvision.transforms as transforms
import timm
import torch.nn as nn

def load_model():
    model = timm.create_model("vit_base_patch16_384", pretrained=False)
    model.head = nn.Linear(model.head.in_features, 10)
    ckpt_path = "cifar-10/models/vit_timm-cifar10-acc-97_47-ckpt.t7"
    checkpoint = torch.load(ckpt_path, map_location=torch.device('cpu'))

    def remove_module_prefix(state_dict):
        return {k[len('module.'):] if k.startswith('module.') else k: v for k, v in state_dict.items()}

    state_dict = remove_module_prefix(checkpoint['model'])
    model.load_state_dict(state_dict)
    return model

def get_transform():
    return transforms.Compose([
        transforms.Resize(384),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

def predict(model, image):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    transform = get_transform()
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)

    # classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    # confidences = {classes[i]: probabilities[i].item() for i in range(len(classes))}

    return probabilities

# Example usage
if __name__ == "__main__":
    from PIL import Image

    model = load_model()

    # Replace 'path/to/your/image.jpg' with the actual path to your image
    image_path = 'cifar-10/ices/0/2.png'
    image = Image.open(image_path).convert('RGB')

    result = predict(model, image)

    print(result)