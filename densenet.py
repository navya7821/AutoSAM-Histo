import torch
from torchvision import models, transforms
from torchvision.models import densenet121
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamPredictor
import json
import urllib

# --- 1. Model Setup ---
model_type = "vit_b"
checkpoint_path = r"C:\Users\lenovo\Desktop\MP\Models\sam_vit_b_01ec64.pth"  # Update this path

sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
predictor = SamPredictor(sam)
print("SAM model loaded")

# Use DenseNet121 pre-trained model (more sensitive for medical images)
from torchvision.models import DenseNet121_Weights

# Use the recommended way to load weights
densenet_model = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)  # Using the updated method to load weights

densenet_model.eval()  # Set to evaluation mode
print("DenseNet121 model loaded")

# --- 2. Load & Preprocess Image ---
image_path = r"C:\Users\lenovo\Desktop\MP\Images\15.png"   # Update this path
image = cv2.imread(image_path)
if image is None:
    raise FileNotFoundError(f"Image not found at {image_path}")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
predictor.set_image(image)
print("Image loaded")

# --- 3. Blob Detection ---
gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
_, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY_INV)

kernel = np.ones((3, 3), np.uint8)
cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
merged = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=2)

contours, _ = cv2.findContours(merged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

input_points = []
for cnt in contours:
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)
    if perimeter == 0:
        continue
    circularity = 4 * np.pi * area / (perimeter ** 2)
    if (50 < area < 5000) and (0.7 < circularity < 1.3):
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            input_points.append([cx, cy])

print(f"Detected {len(input_points)} input points for segmentation.")

if len(input_points) == 0:
    print("No valid input points detected for segmentation. Skipping segmentation.")
    masks = []
else:
    input_points = np.array(input_points)
    input_labels = np.ones(len(input_points))  # All points labeled positive for SAM
    masks, _, _ = predictor.predict(
        point_coords=input_points,
        point_labels=input_labels,
        multimask_output=False,
    )
    print("Segmentation done")

# --- 4. Define Image Transformation ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resizing the image to match the input size of DenseNet121
    transforms.ToTensor(),  # Convert the image to a tensor
    transforms.Normalize(   # Normalize the image to match DenseNet121's training data
        mean=[0.485, 0.456, 0.406],  # ImageNet's normalization values
        std=[0.229, 0.224, 0.225]
    )
])

# --- 5. Classifier: Predict Patch ---
def predict_patch(patch_pil):
    input_tensor = transform(patch_pil).unsqueeze(0)
    with torch.no_grad():
        output = densenet_model(input_tensor)
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    return probabilities

# --- 6. Fetch ImageNet Class Labels ---
# Download ImageNet labels from a reliable source
LABELS_URL = 'https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json'
with urllib.request.urlopen(LABELS_URL) as url:
    class_idx = json.load(url)


imagenet_classes = [class_idx[str(k)][1] for k in range(1000)]


predictions = []

CONFIDENCE_THRESHOLD = 0.5  

if len(masks) > 0:
    for mask in masks:
        ys, xs = np.where(mask)
        if len(xs) == 0 or len(ys) == 0:
            continue
        x_min, x_max = xs.min(), xs.max()
        y_min, y_max = ys.min(), ys.max()

        patch = image[y_min:y_max, x_min:x_max]
        patch_pil = Image.fromarray(patch)

        probs = predict_patch(patch_pil)
        top_prob, top_class = torch.max(probs, 0)
        class_index = top_class.item()

        # Fetch human-readable label for the predicted class
        label = imagenet_classes[class_index]
        confidence = top_prob.item()
        if confidence >= CONFIDENCE_THRESHOLD:
            predictions.append((label, confidence))
        else:
            predictions.append(("Uncertain", confidence))  # Mark as uncertain if confidence is too low
    print("Classification done")
else:
    print("No masks to classify.")

# --- 8. Visualization ---
plt.figure(figsize=(15, 7))

# Original Image
plt.subplot(1, 2, 1)
plt.imshow(image)
plt.title("Original Image")
plt.axis('off')

# Segmentation + Classification
plt.subplot(1, 2, 2)
plt.imshow(image)
plt.axis('off')

if len(masks) > 0:
    for i, mask in enumerate(masks):
        plt.imshow(mask, alpha=0.5, cmap='jet')
        if i < len(predictions):
            label, conf = predictions[i]
            text = f"{label} ({conf*100:.1f}%)" if conf >= CONFIDENCE_THRESHOLD else f"Uncertain ({conf*100:.1f}%)"
            plt.text(
                x=10, y=30 + i * 20,
                s=text,
                color='white' if conf >= CONFIDENCE_THRESHOLD else 'yellow',
                fontsize=12,
                backgroundcolor='black',
                bbox=dict(facecolor='black', alpha=0.6, pad=2)
            )

plt.title("Segmentation + Classification")
plt.show()

print("Program executed successfully")