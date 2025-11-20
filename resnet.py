import cv2
import numpy as np
import torch
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamPredictor
from torchvision.models import ResNet50_Weights

# --- 1. Model Setup ---
model_type = "vit_b"  # Use "vit_b" for SAM model
checkpoint_path = r"C:\Users\lenovo\Desktop\MP\Models\sam_vit_b_01ec64.pth"  # Update this path

sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
predictor = SamPredictor(sam)
print("SAM model loaded")

# Load the ResNet50 pretrained model with updated weights method
resnet_model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
resnet_model.eval()  # Set to evaluation mode
print("ResNet50 model loaded")

# Get the human-readable class labels from ImageNet categories
imagenet_classes = ResNet50_Weights.IMAGENET1K_V1.meta['categories']

# Define the image transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# --- 2. Load & Preprocess Image ---

image_path = r"C:\Users\lenovo\Desktop\MP\Images\1.png"  # Update this path
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

# --- 4. Classifier: Predict Patch ---
def predict_patch(patch_pil):
    input_tensor = transform(patch_pil).unsqueeze(0)
    with torch.no_grad():
        output = resnet_model(input_tensor)
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    return probabilities

# --- 5. Extract patches from masks and classify ---
predictions = []
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
        
        # Map the class index to the human-readable label
        label = imagenet_classes[class_index] if class_index < len(imagenet_classes) else "Other"
        
        confidence = top_prob.item()
        predictions.append((label, confidence))
    print("Classification done")
else:
    print("No masks to classify.")

# --- 6. Visualization ---
CONFIDENCE_THRESHOLD = 0.5  # 50%

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

print("Program finished!")


import cv2
import numpy as np
import os
from segment_anything import sam_model_registry, SamPredictor

# ------------------ SETTINGS ------------------
MODEL_TYPE = "vit_b"
CHECKPOINT_PATH = r"C:\Users\lenovo\Desktop\MP\Models\sam_vit_b_01ec64.pth"
IMAGE_FOLDER = r"C:\Users\lenovo\Desktop\MP\Images"

MIN_AREA = 50
MAX_AREA = 5000

# ------------------ LOAD SAM ------------------
sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH)
predictor = SamPredictor(sam)
print("SAM loaded")

# ------------------ HELPER FUNCTIONS ------------------
def detect_blobs_otsu(gray):
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = np.ones((3,3), np.uint8)
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    merged = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(merged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    points = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        circularity = 4 * np.pi * area / (perimeter**2 + 1e-6)
        if MIN_AREA < area < MAX_AREA and 0.5 < circularity < 1.5:
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"]/M["m00"])
                cy = int(M["m01"]/M["m00"])
                points.append((cx, cy))
    return points

def detect_blobs_hsv(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    lower = np.array([0, 50, 50])
    upper = np.array([179, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)
    kernel = np.ones((3,3), np.uint8)
    cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    points = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if MIN_AREA < area < MAX_AREA:
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"]/M["m00"])
                cy = int(M["m01"]/M["m00"])
                points.append((cx, cy))
    return points

def get_fallback_points(image):
    h, w = image.shape[:2]
    return [(w//2, h//2), (w//4, h//4), (3*w//4, h//4), (w//4, 3*h//4), (3*w//4, 3*h//4)]

# ------------------ MAIN LOOP ------------------
segmented_images = {}  # dictionary to store masks in-memory

for img_name in os.listdir(IMAGE_FOLDER):
    img_path = os.path.join(IMAGE_FOLDER, img_name)
    image = cv2.imread(img_path)
    if image is None:
        print(f"Skipping {img_name}, could not read")
        continue
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    predictor.set_image(image)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Multi-strategy blob detection
    points_otsu = detect_blobs_otsu(gray)
    points_hsv = detect_blobs_hsv(image)
    candidate_points = points_otsu + points_hsv

    # Fallback if no points found
    if len(candidate_points) == 0:
        candidate_points = get_fallback_points(image)

    input_points = np.array(candidate_points)
    input_labels = np.ones(len(input_points))

    masks, _, _ = predictor.predict(
        point_coords=input_points,
        point_labels=input_labels,
        multimask_output=False,
    )

    if masks is not None and len(masks) > 0:
        mask = masks[0].astype(np.uint8)
        segmented_images[img_name] = mask  # store in-memory for training
    else:
        # fallback mask: full image
        mask = np.ones(image.shape[:2], dtype=np.uint8)
        segmented_images[img_name] = mask

print(f"Segmentation done for {len(segmented_images)} images. Ready for scaling-law pipeline.")