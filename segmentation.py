import cv2
import numpy as np
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamPredictor

# --- 1. Model Setup ---
model_type = "vit_b"
checkpoint_path = r"C:\Users\lenovo\Desktop\MP\Models\sam_vit_b_01ec64.pth"  
sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
predictor = SamPredictor(sam)
print("Model loaded")

# --- 2. Load & Preprocess Image ---
image_path = r"C:\Users\lenovo\Desktop\MP\Images\3.png"  
image = cv2.imread(image_path)
if image is None:
    raise FileNotFoundError(f"{image} not found")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
predictor.set_image(image)
print("Image loaded")

# --- 3. Optimized Blob Detection ---
gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

# Otsu's thresholding + binary inversion
_, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY_INV)

# Morphological improvements
kernel = np.ones((3,3), np.uint8)
cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)  
merged = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=2)  

# Contour detection with shape/size filters
contours, _ = cv2.findContours(merged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
input_points = []
for cnt in contours:
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)
    circularity = 4 * np.pi * area / (perimeter**2)
    
    if (50 < area < 5000) and (0.7 < circularity < 1.3):  
        M = cv2.moments(cnt)
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        input_points.append([cx, cy])
print("Blob done")

# --- 4. SAM Segmentation ---
input_points = np.array(input_points)
input_labels = np.ones(len(input_points))
masks, _, _ = predictor.predict(
    point_coords=input_points,
    point_labels=input_labels,
    multimask_output=False,
)
print("Segmentation done")
# --- 5. Visualization ---
plt.figure(figsize=(15, 7))
plt.subplot(1, 2, 1)
plt.imshow(image)
plt.title("Original Image")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(image)
plt.imshow(masks[0], alpha=0.5, cmap='jet')
plt.title("Automatic Segmentation Result")
plt.axis('off')
plt.show()
print("Program executed successfully")