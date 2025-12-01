"""
CNN Image Classification Page
Crop Disease and Soil Moisture Classification using CNN/ResNet50 models
"""

import os
import sys
from pathlib import Path
import streamlit as st
from PIL import Image
from collections import defaultdict
import random

# Prevent segmentation faults with PyTorch
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'

# Add parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
# Go up from pages/ to streamlit_app/ to project root
streamlit_app_dir = os.path.dirname(current_dir)
parent_dir = os.path.dirname(streamlit_app_dir)
# Add both paths
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
if streamlit_app_dir not in sys.path:
    sys.path.insert(0, streamlit_app_dir)

# Add Final_Project to path
final_project_dir = os.path.join(parent_dir, "Final_Project")
if final_project_dir not in sys.path:
    sys.path.insert(0, final_project_dir)

# Import with error handling to avoid segfault
_disease_module_loaded = False
_soil_module_loaded = False

try:
    # Only import functions, not load models at import time
    import Disease_classification
    import Soil_Classification
    _disease_module_loaded = True
    _soil_module_loaded = True
    get_sample_images = Disease_classification.get_sample_images
    predict_disease_image = Disease_classification.predict_image
    predict_soil_image = Soil_Classification.predict_image
except ImportError as e:
    st.error(f"❌ Error importing CNN modules: {str(e)}")
    st.info("Please ensure Final_Project folder contains Disease_classification.py and Soil_Classification.py")
    st.stop()
except Exception as e:
    st.warning(f"⚠️ Warning during import: {str(e)}")
    st.info("Models will be loaded when needed. This is normal.")
    # Try to import anyway
    try:
        import Disease_classification
        import Soil_Classification
        _disease_module_loaded = True
        _soil_module_loaded = True
        get_sample_images = Disease_classification.get_sample_images
        predict_disease_image = Disease_classification.predict_image
        predict_soil_image = Soil_Classification.predict_image
    except:
        st.error("Failed to import CNN modules. Please check the files.")
        st.stop()

st.title("🖼️ CNN Image Classification")

# Disease class labels
disease_labels = [
    'American Bollworm on Cotton', 'Anthracnose on Cotton', 'Army worm',
    'Becterial Blight in Rice', 'Brownspot', 'Common_Rust',
    'Cotton Aphid', 'Flag Smut', 'Gray_Leaf_Spot',
    'Healthy Maize', 'Healthy Wheat', 'Healthy cotton',
    'Leaf Curl', 'Leaf smut', 'Mosaic sugarcane',
    'RedRot sugarcane', 'RedRust sugarcane', 'Rice Blast',
    'Sugarcane Healthy', 'Tungro', 'Wheat Brown leaf Rust',
    'Wheat Stem fly', 'Wheat aphid', 'Wheat black rust',
    'Wheat leaf blight', 'Wheat mite', 'Wheat powdery mildew',
    'Wheat scab', 'Wheat___Yellow_Rust', 'Wilt',
    'Yellow Rust Sugarcane', 'bacterial_blight in Cotton',
    'bollworm on Cotton', 'cotton mealy bug',
    'cotton whitefly', 'maize ear rot', 'maize fall armyworm',
    'maize stem borer', 'pink bollworm in cotton',
    'red cotton bug', 'thirps on  cotton'
]

# Moisture labels
moisture_labels = ['0-10', '11-20', '21-40', '41-100']

# Model paths (relative to Final_Project)
final_project_path = Path(parent_dir) / "Final_Project"
cnn_soil_path = final_project_path / "cnn_soil_best.pth"
resnet_soil_path = final_project_path / "resnet50_soil_best.pth"
cnn_plant_path = final_project_path / "CNN_Plant_best.pth"
resnet_plant_path = final_project_path / "ResNet_Plant_best.pth"

# Dataset paths
dataset_root = final_project_path / "CropDiseaseImages" / "Validation"

# Tabs for Disease & Soil Moisture
tab1, tab2 = st.tabs(["🌾 Crop Disease Classification", "💧 Soil Moisture Classification"])

# =======================================================
# TAB 1 – CROP DISEASE CLASSIFICATION
# =======================================================
with tab1:
    st.header("🌾 Crop Disease Classification")
    
    uploaded_disease = st.file_uploader(
        "Upload an image for disease prediction",
        type=["jpg", "jpeg", "png"],
        key="disease_upload"
    )
    
    model_choice = st.selectbox("Select Model", ["CNN", "ResNet50"], key="disease_model_select")
    confidence_threshold = st.slider("Confidence Threshold (%)", 0, 100, 50, key="confidence_slider")
    
    actual_label = st.selectbox(
        "Actual disease label (optional):",
        ["None"] + disease_labels,
        key="actual_label_select"
    )
    if actual_label == "None":
        actual_label = None
    
    if uploaded_disease and st.button("Predict Disease", key="predict_disease_btn"):
        try:
            with st.spinner("Loading model and making prediction..."):
                predicted_class, confidence, image_obj = predict_disease_image(
                    model_type=model_choice.lower(),
                    img_file=uploaded_disease
                )
            
            col1, col2 = st.columns(2)
            with col1:
                st.image(image_obj, caption="Uploaded Image", use_container_width=True)
            with col2:
                st.subheader("Prediction Result")
                st.write(f"**Predicted Class:** {predicted_class}")
                st.write(f"**Confidence:** {confidence:.2f}%")
                st.write(f"**Model Used:** {model_choice}")
                if actual_label:
                    st.write(f"**Actual Label:** {actual_label}")
                
                if confidence < confidence_threshold:
                    st.warning("Prediction confidence below threshold!")
                else:
                    st.success("Prediction confidence meets threshold!")
        except RuntimeError as e:
            st.error(f"Model loading error: {str(e)}")
            st.info("This might be a segmentation fault. Try restarting the app or check model files.")
        except Exception as e:
            st.error(f"Error during prediction: {str(e)}")
            st.info("Please check that model files exist in Final_Project folder")
            import traceback
            st.code(traceback.format_exc())
    
    # Sample images section
    st.markdown("---")
    st.subheader("📌 View Sample Crop Disease Images")
    
    num_classes = st.number_input(
        "Number of classes to display:",
        min_value=1, max_value=len(disease_labels), value=5,
        key="num_classes_input"
    )
    
    num_samples = st.number_input(
        "Number of sample images per class:",
        min_value=1, max_value=42, value=2,
        key="num_samples_input"
    )
    
    if st.button("Show Sample Images", key="show_samples_btn"):
        if not dataset_root.exists():
            st.warning(f"Dataset path not found: {dataset_root}")
        else:
            try:
                samples = get_sample_images(str(dataset_root), num_samples_per_class=num_samples)
                
                if not samples:
                    st.warning("No images found. Check dataset path and folder names!")
                else:
                    st.success(f"Displaying sample images for {num_classes} classes (max {num_samples} images per class).")
                    
                    # Organize images by class
                    class_dict = defaultdict(list)
                    for img, label in samples:
                        if len(class_dict[label]) < num_samples:
                            class_dict[label].append(img)
                    
                    # Limit to requested number of classes
                    available_classes = list(class_dict.keys())
                    if len(available_classes) > num_classes:
                        selected_classes = random.sample(available_classes, num_classes)
                    else:
                        selected_classes = available_classes
                    
                    # Display each class separately
                    for class_name in selected_classes:
                        images = class_dict[class_name]
                        st.subheader(f"{class_name} ({len(images)} sample{'s' if len(images)>1 else ''})")
                        cols_per_row = min(5, len(images))
                        rows = (len(images) + cols_per_row - 1) // cols_per_row
                        
                        for r in range(rows):
                            cols = st.columns(cols_per_row)
                            for c in range(cols_per_row):
                                idx = r * cols_per_row + c
                                if idx < len(images):
                                    cols[c].image(images[idx], use_container_width=True)
            except Exception as e:
                st.error(f"Error loading sample images: {str(e)}")

# =======================================================
# TAB 2 – SOIL MOISTURE CLASSIFICATION
# =======================================================
with tab2:
    st.header("💧 Soil Moisture Classification")
    
    uploaded_soil = st.file_uploader(
        "Upload an image for soil moisture classification",
        type=["jpg", "jpeg", "png"],
        key="soil_upload"
    )
    
    actual_label_soil = st.selectbox(
        "Actual moisture label (optional):",
        ["None"] + moisture_labels,
        key="actual_label_soil_select"
    )
    if actual_label_soil == "None":
        actual_label_soil = None
    
    model_choice_soil = st.selectbox(
        "Select Moisture Model:",
        ["CNN", "ResNet50"],
        key="soil_model_select"
    )
    
    # Define checkpoint paths
    checkpoint_path = cnn_soil_path if model_choice_soil == "CNN" else resnet_soil_path
    
    if uploaded_soil and st.button("Predict Soil Moisture", key="predict_soil_btn"):
        if not checkpoint_path.exists():
            st.error(f"Model file not found: {checkpoint_path}")
            st.info("Please ensure model files exist in Final_Project folder")
        else:
            try:
                with st.spinner("Loading model and making prediction..."):
                    predicted_class, confidence, image_obj = predict_soil_image(
                        model_path=str(checkpoint_path),
                        image_path=uploaded_soil,
                        class_names=moisture_labels,
                        model_type=model_choice_soil.lower()
                    )
                
                col1, col2 = st.columns(2)
                with col1:
                    st.image(image_obj, caption="Uploaded Soil Image", use_container_width=True)
                with col2:
                    st.subheader("Moisture Prediction Result")
                    st.write(f"**Predicted Moisture Range:** {predicted_class}")
                    st.write(f"**Confidence:** {confidence:.2f}%")
                    st.write(f"**Model Used:** {model_choice_soil}")
                    if actual_label_soil:
                        st.write(f"**Actual Label:** {actual_label_soil}")
                    
                    if confidence < 50:
                        st.warning("Confidence is low!")
                    else:
                        st.success("Reliable moisture prediction!")
            except RuntimeError as e:
                st.error(f"Model loading error: {str(e)}")
                st.info("This might be a segmentation fault. Try restarting the app or check model files.")
            except Exception as e:
                st.error(f"Error during prediction: {str(e)}")
                import traceback
                st.code(traceback.format_exc())

