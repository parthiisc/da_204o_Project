# Integration Guide: CNN App with Soil Moisture Prediction

## Overview

The CNN Image Classification application has been successfully integrated with the Soil Moisture Prediction app into a unified multi-page Streamlit application with white background styling.

## Structure

```
streamlit_app/
├── app.py                                    # Main home page (white background)
├── pages/
│   ├── 1_🌾_Soil_Moisture_Prediction.py      # Original ML prediction app
│   └── 2_🖼️_CNN_Image_Classification.py     # CNN classification app
└── [other modules...]

Final_Project/
├── Disease_classification.py                 # Updated with proper paths
├── Soil_Classification.py                   # Already uses parameter paths
├── CNN_Plant_best.pth                        # Disease classification models
├── ResNet_Plant_best.pth
├── cnn_soil_best.pth                         # Soil moisture models
├── resnet50_soil_best.pth
└── CropDiseaseImages/                         # Dataset
```

## Changes Made

### 1. Main App (app.py)
- **Converted to home page** with white background styling
- **Custom CSS** for white background (#FFFFFF)
- **Application overview** cards
- **Quick statistics** section
- **Navigation instructions**

### 2. Multi-Page Structure
- **Page 1**: Soil Moisture Prediction (moved from app.py)
- **Page 2**: CNN Image Classification (integrated from Final_Project/Streamlit.py)

### 3. Path Fixes
- **Disease_classification.py**: Updated to use `Path(__file__).parent` for model paths
- **CNN page**: Uses relative paths from Final_Project directory
- **All Windows paths removed**: Now uses cross-platform Path objects

### 4. White Background Styling
- Main background: #FFFFFF
- Sidebar: #F8F9FA
- Custom CSS for headers, buttons, and info boxes
- Green color scheme for agricultural theme

## How to Run

```bash
cd /Users/parthpatel/Mtech/DA_204o_Project
streamlit run streamlit_app/app.py
```

The app will open with:
1. **Home page** showing overview
2. **Sidebar navigation** to access both applications
3. **White background** throughout

## Features

### Home Page
- Welcome message
- Application overview cards
- Quick statistics
- Getting started guide

### Soil Moisture Prediction Page
- All original features preserved
- ML model predictions (XGBoost, LightGBM)
- District rankings
- 3-month projections
- Crop recommendations
- Visualizations

### CNN Image Classification Page
- **Crop Disease Classification**
  - Upload image
  - Choose model (CNN/ResNet50)
  - View predictions with confidence
  - Sample image gallery
  
- **Soil Moisture Classification**
  - Upload soil image
  - Choose model (CNN/ResNet50)
  - Get moisture range prediction
  - Confidence scoring

## Requirements

Ensure these files exist in `Final_Project/`:
- `CNN_Plant_best.pth` - Disease CNN model
- `ResNet_Plant_best.pth` - Disease ResNet model
- `cnn_soil_best.pth` - Soil moisture CNN model
- `resnet50_soil_best.pth` - Soil moisture ResNet model
- `CropDiseaseImages/Validation/` - Dataset for sample images

## Troubleshooting

### Models Not Found
If you see "Model file not found" errors:
1. Check that model files exist in `Final_Project/` directory
2. Verify file names match exactly (case-sensitive)
3. Check file permissions

### Import Errors
If you see import errors:
1. Ensure `Final_Project/` is in the project root
2. Check that `Disease_classification.py` and `Soil_Classification.py` exist
3. Verify Python dependencies are installed (torch, torchvision, PIL)

### Path Issues
All paths are now relative to the script locations. If issues persist:
1. Check that `Final_Project/` directory structure is correct
2. Verify model files are in the right location
3. Check dataset path for sample images

## Next Steps

1. **Test both applications** to ensure everything works
2. **Verify model files** are in correct locations
3. **Test image uploads** for CNN classification
4. **Check white background** displays correctly
5. **Test navigation** between pages

## Notes

- The original `app.py` functionality is preserved in `pages/1_🌾_Soil_Moisture_Prediction.py`
- CNN app is integrated in `pages/2_🖼️_CNN_Image_Classification.py`
- All hardcoded Windows paths have been replaced with cross-platform Path objects
- White background styling is applied globally via custom CSS

---

**Last Updated:** [Current Date]

