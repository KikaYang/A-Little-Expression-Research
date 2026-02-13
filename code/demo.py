import torch
import torch.nn as nn
from PIL import Image
import gradio as gr
from torchvision import models
from torchvision.models import ResNet18_Weights

# -----------------------
# Config
# -----------------------
CHECKPOINT_PATH = "/Users/yangtianchi/Downloads/best_resnet18.pth"   
NUM_CLASSES = 3                        
CLASS_NAMES = ["happy", "neutral", "sad"] 

# -----------------------
# Device
# -----------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------
# Build model
# -----------------------
weights = ResNet18_Weights.DEFAULT
preprocess = weights.transforms()  

model = models.resnet18(weights=weights)
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)

# load checkpoint
state = torch.load(CHECKPOINT_PATH, map_location=device)
model.load_state_dict(state)

model.to(device)
model.eval()

# -----------------------
# Predict function
# -----------------------

@torch.inference_mode()
def predict(img: Image.Image):
    img = img.convert("RGB")

    x = preprocess(img).unsqueeze(0).to(device)  # [1,3,224,224]
    logits = model(x)
    probs = torch.softmax(logits, dim=1).squeeze(0).cpu()

    pred_idx = int(torch.argmax(probs).item())
    pred_label = CLASS_NAMES[pred_idx] if pred_idx < len(CLASS_NAMES) else str(pred_idx)

    # return dict: label->confidence
    result = {CLASS_NAMES[i]: float(probs[i]) for i in range(min(len(CLASS_NAMES), probs.shape[0]))}

    return pred_label, result


# -----------------------
# Gradio UI
# -----------------------
demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload an image"),
    outputs=[
        gr.Textbox(label="Prediction"),
        gr.Label(label="Probabilities")
    ],
    title="Expression Detection (ResNet18)",
    description="Upload an image and get the predicted expression class.",
)

if __name__ == "__main__":
    demo.launch()
