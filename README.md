# Automatic Segmentation using SAM (Segment Anything Model)

**Zero-click, fully automatic histopathology image segmentation**  
Core Innovation: Automatic point-prompt generation for SAM using **Otsu’s binary thresholding + morphological operations + contour filtering**

The objective of this project is to eliminate all manual interaction with the Segment Anything Model (SAM) by building a classic computer-vision preprocessing pipeline that automatically creates high-quality point prompts.  
Otsu’s thresholding → binary inversion → morphological opening & closing → contour detection → area & circularity filtering → centroid extraction → fed directly into SAM as point prompts.

This turns the powerful but interactive SAM into a completely automatic segmentation engine — ideal for large-scale medical image analysis.

Optional classification of each segmented region using pretrained ResNet50 and DenseNet121 (ImageNet weights) is included for interpretability.

### Dataset
BreakHis (Breast Cancer Histopathological Images)  
- 9,109 images (2,480 benign | 5,429 malignant)  
- All magnifications (40×, 100×, 200×, 400×)  
- Resized to 256×256  
- No annotations used — segmentation generated fully automatically

### Core Pipeline (Automatic Prompt Generation)
1. Resize → 256×256  
2. Grayscale conversion  
3. Otsu’s binary thresholding + inversion (highlights foreground tissue)  
4. Morphological opening & closing (removes noise, fills gaps)  
5. Contour detection  
6. Filter contours by area and circularity (removes irrelevant blobs)  
7. Extract centroids of valid contours → used as positive point prompts for SAM  

SAM (ViT-B) then produces clean, high-resolution masks with zero user input.

### Optional Classification
- Bounding-box crop around each mask  
- Resize to 224×224 + ImageNet normalization  
- Classify using ResNet50 or DenseNet121 (pretrained)  
- Confidence ≥ 50% → show ImageNet label | < 50% → “Uncertain”

### Implementation
Three scripts provided:
- `segmentation.py` → pure automatic segmentation  
- `resnet.py` → segmentation + ResNet50 classification  
- `densenet.py` → segmentation + DenseNet121 classification  

### Libraries
torch | torchvision | opencv-python | segment-anything | matplotlib | numpy | Pillow

### Results
- SAM consistently generates sharp, accurate masks that perfectly outline tissue structures  
- No manual clicks or prompts ever required  
- ResNet50 gives stable predictions; DenseNet121 offers richer variety  
- Confidence thresholding prevents over-interpretation

### Conclusion
By combining simple, robust classical CV techniques (Otsu + morphology + contours) with the Segment Anything Model, this project achieves true zero-click, high-quality segmentation on challenging histopathology images. The pipeline is lightweight, fully automatic, and easily extensible to other medical imaging tasks.
