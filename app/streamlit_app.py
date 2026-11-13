
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import cv2
import torch
from torchvision import transforms
from PIL import Image, ImageOps
import streamlit as st

from src.models.classifier import build_classifier
from src.explainability.gradcam import SimpleGradCAM, overlay_heatmap

APP_TITLE = "AirNox"
FOCUS_MIN = 35.0
CONFIDENCE_MIN = 0.70
CHECKPOINT = ROOT / "models" / "checkpoints" / "airnox_resnet18_best.pt"

st.set_page_config(page_title="AirNox Research", page_icon="🔬", layout="wide")

# Theme
st.markdown("""
<style>
.stApp {
    background: #0D1713;
    color: #E7F0E9;
}
.block-container {
    max-width: 1450px;
    padding: 2rem 3rem 3rem;
}
h1, h2, h3 {
    color: #E8F1EA !important;
    letter-spacing: -0.7px;
}
p, label, .stMarkdown { color: #B0C2B5; }

[data-testid="stSidebar"] {
    background: #14231B;
    border-right: 1px solid #293D30;
}
[data-testid="stSidebar"] * { color: #DCE9DF; }

[data-testid="stVerticalBlockBorderWrapper"] {
    background: #17261E;
    border: 1px solid #304638;
    border-radius: 18px;
    padding: 18px;
}

[data-testid="stMetric"] {
    background: #203329;
    border: 1px solid #385543;
    border-radius: 13px;
    padding: 16px;
}
[data-testid="stMetricLabel"] { color: #A7BDAE !important; }
[data-testid="stMetricValue"] { color: #D8EBDD !important; }

.stButton > button {
    background: #315B40;
    color: white;
    border: 1px solid #52785D;
    border-radius: 10px;
}
.stButton > button:hover {
    background: #416F50;
    color: white;
    border-color: #82B494;
}

[data-testid="stFileUploader"] {
    background: #1C2D23;
    border: 1px dashed #587663;
    border-radius: 12px;
    padding: 12px;
}

[data-testid="stImage"] img { border-radius: 12px; }

.stProgress > div > div > div {
    background: #82B494;
}
[data-testid="stAlert"] {
    border-radius: 12px;
}
hr { border-color: #304638 !important; }

[data-testid="stCaptionContainer"] { color: #82998B; }

#MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# Database
ALLERGEN_REFERENCE_DB = {
    "birch_pollen": {
        "scientific_name": "Betula spp. (Birch)",
        "morphology": "20–28 µm triporate grain with smooth exine and three germinal pores.",
        "peak_season": "Spring (April–May)",
        "allergenicity": "High. Major allergen Bet v 1.",
        "exposure_guidance": "Common near birch stands on warm, dry, breezy mornings."
    },
    "ragweed_pollen": {
        "scientific_name": "Ambrosia artemisiifolia",
        "morphology": "18–22 µm spheroidal grain with short conical spines.",
        "peak_season": "Late summer–autumn (August–October)",
        "allergenicity": "Very high. Major allergen Amb a 1.",
        "exposure_guidance": "Airborne concentrations can rise during dry, windy conditions."
    },
    "grass_pollen": {
        "scientific_name": "Poaceae family",
        "morphology": "35–55 µm ovoid to spheroidal grain with a single pore.",
        "peak_season": "Late spring–mid summer (May–July)",
        "allergenicity": "High. Includes Phl p 1 and Phl p 5.",
        "exposure_guidance": "Concentrations can rise during lawn mowing and dry afternoons."
    },
    "mold_spores": {
        "scientific_name": "Alternaria and Cladosporium spp.",
        "morphology": "Dark pigmented spores, often elongated with septa.",
        "peak_season": "Summer through late fall; year-round indoors.",
        "allergenicity": "High. Some fungal spores are associated with asthma exacerbations.",
        "exposure_guidance": "Common near damp soil, decaying leaves, and poorly ventilated areas."
    },
    "benign_dust": {
        "scientific_name": "Non-biological particulate",
        "morphology": "Irregular angular mineral or carbonaceous fragments.",
        "peak_season": "Year-round.",
        "allergenicity": "Not a conventional IgE-mediated allergen; may irritate airways.",
        "exposure_guidance": "Associated with traffic, construction, and dry environments."
    }
}

# Model
def compute_focus_score(img):
    """
    Measures image sharpness using Laplacian variance. Higher scores-->sharper images, while lower scores may indicate blur or poor microscope focus.
    """
    gray = cv2.cvtColor(
        np.array(img.resize((224, 224))), cv2.COLOR_RGB2GRAY
    )
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())

@st.cache_resource
def load_trained_model():
    """
    Loads the trained ResNet classifier from the saved checkpoint.
    Restores model weights and class labels, then switches the model to evaluation mode for inference.
    """
    if not CHECKPOINT.exists():
        return None, None
        
    ckpt = torch.load(CHECKPOINT, map_location="cpu")
    classes = ckpt["classes"]
    model = build_classifier(
        arch=ckpt["arch"], num_classes=len(classes), pretrained=False
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, classes


@st.cache_resource
def load_gradcam(_model):
    """
    Initializes Grad-CAM for visualizing model attention.Uses the final ResNet18 layer to generate heatmaps highlighting image regions influencing predictions.
    """
    return SimpleGradCAM(_model, _model.layer4[-1])


# Header
st.markdown("""
<div style="display:flex;align-items:center;gap:14px;margin-bottom:8px">
    <div style="background:#244634;padding:13px;border-radius:14px;
                font-size:25px">🔬</div>
    <div>
        <div style="font-size:30px;font-weight:700;color:#E8F1EA;
                    letter-spacing:-1px">AirNox</div>
        <div style="font-size:11px;letter-spacing:2px;color:#86AA91">
            BIOAEROSOL RESEARCH PLATFORM
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown(
    "Microscopic particle classification, model explainability, "
    "and environmental exposure research."
)

st.divider()

model, classes = load_trained_model()

if model is None:
    st.error("Model checkpoint not found. Run python -m src.training.train first.")
    st.stop()

# Sidebar
with st.sidebar:
    st.markdown("## 🔬 Research Console")
    st.caption("AIRNOX / MVP 01")
    st.divider()

    st.markdown("### System Overview")
    st.markdown("""
    **Architecture**  
    ResNet18

    **Explainability**  
    Grad-CAM

    **Quality Control**  
    Laplacian variance

    **Inference**  
    PyTorch
    """)

    st.divider()
    st.markdown("### Research Limitations")
    st.caption(
        "Predictions may vary across microscope sources, "
        "staining techniques, and image quality. "
        "Low-confidence predictions require expert verification."
    )
    st.caption("Environmental research only. Not a medical device.")

# Main UI component
left, right = st.columns([1, 1.15], gap="large")

selected_image = None
true_label_hint = None
focus_var = None

with left:
    with st.container(border=True):
        st.markdown("### 01 / Sample Acquisition")
        st.caption("Select a microscopy sample for analysis.")

        # Check if local test data exists-->safe for Streamlit Cloud environments
        test_dir = ROOT / "data" / "processed" / "test"
        assets_dir = ROOT / "assets"

        sample_files = []
        if test_dir.exists() and any(test_dir.glob("*/*.jpg")):
            sample_files = sorted(test_dir.glob("*/*.jpg"))
            has_test_data = True
        elif assets_dir.exists() and any(assets_dir.glob("**/*.jpg")):
            sample_files = sorted(assets_dir.glob("**/*.jpg"))
            has_test_data = True
        else:
            has_test_data = False

        default_mode_idx = 0 if has_test_data else 1
        
        mode = st.radio(
            "Input source",
            ["Test dataset", "Upload image"],
            index=default_mode_idx,
            horizontal=True
        )

        if mode == "Test dataset" and has_test_data:
            choice = st.selectbox(
                "Select sample",
                sample_files,
                format_func=lambda p: f"{p.parent.name} / {p.name}" if p.parent.name != "assets" else f"Demo Sample / {p.name}"
            )
            selected_image = Image.open(choice).convert("RGB")
            true_label_hint = choice.parent.name if choice.parent.name != "assets" else "Demo Sample"
        else:
            if mode == "Test dataset" and not has_test_data:
                st.info("No demo samples found in cloud environment. Please upload an image.")
            
            uploaded = st.file_uploader(
                "Upload microscope image",
                type=["jpg", "jpeg", "png"]
            )
            if uploaded:
                selected_image = Image.open(uploaded).convert("RGB")

# Results
with right:
    with st.container(border=True):
        st.markdown("### 03 / Model Inference")
        st.caption("Neural classification and prediction confidence")

        if selected_image is None:
            st.info("Select or upload a sample to begin analysis.")

        else:
            norm_img = ImageOps.autocontrast(
                selected_image, cutoff=1
            ).resize((224, 224))

            tf = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.485, 0.456, 0.406],
                    [0.229, 0.224, 0.225]
                )
            ])

            tensor = tf(norm_img).unsqueeze(0)

            with torch.no_grad():
                probs = torch.softmax(model(tensor), dim=1)[0].numpy()

            top_idx = int(np.argmax(probs))
            pred_class = classes[top_idx]
            confidence = float(probs[top_idx])

            c1, c2 = st.columns(2)
            c1.metric(
                "Predicted Class",
                pred_class.replace("_", " ").title()
            )
            c2.metric("Confidence", f"{confidence * 100:.1f}%")

            if confidence < CONFIDENCE_MIN:
                st.warning("Low confidence. Expert verification recommended.")
            else:
                st.success("Prediction confidence above configured threshold.")

            if true_label_hint:
                st.caption(f"Test dataset label: {true_label_hint}")

            st.markdown("#### Class Probability Distribution")

            for idx, cname in enumerate(classes):
                st.progress(
                    float(probs[idx]),
                    text=f"{cname.replace('_', ' ').title()} — {probs[idx]*100:.1f}%"
                )

    if selected_image is not None:
        with st.container(border=True):
            st.markdown("### 04 / Explainability")
            st.caption("Grad-CAM attention visualization")

            cam_mask = load_gradcam(model).generate(tensor, top_idx)
            cam_overlay = overlay_heatmap(norm_img, cam_mask)

            st.image(
                cam_overlay,
                caption="Grad-CAM: regions influencing the prediction",
                use_container_width=True
            )

        with st.container(border=True):
            st.markdown("### 05 / Scientific Reference Library")
            st.caption("Environmental exposure context")

            ref = ALLERGEN_REFERENCE_DB.get(pred_class)

            if ref:
                st.markdown(f"**Taxon / Category**")
                st.write(ref["scientific_name"])

                st.markdown("**Morphology**")
                st.write(ref["morphology"])

                st.markdown("**Typical Peak Season**")
                st.write(ref["peak_season"])

                st.markdown("**Documented Allergenicity**")
                st.write(ref["allergenicity"])

                st.markdown("**Environmental Context**")
                st.write(ref["exposure_guidance"])
            else:
                st.info("No reference available for this class.")

# footer
st.divider()
st.markdown("""
<div style="text-align:center;padding:15px;color:#718A79;
font-size:11px;letter-spacing:1.5px">
AIRNOX · BIOAEROSOL RESEARCH · ENVIRONMENTAL EXPOSURE INFORMATION ONLY
<br><br>
NOT A MEDICAL DIAGNOSTIC DEVICE
</div>
""", unsafe_allow_html=True)
