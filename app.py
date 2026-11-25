
# ============================================================
# EXPLAINABLE BRAIN TUMOR DETECTION
# Vision Transformer with Native Multi-Head Self-Attention
# ============================================================

import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras import layers
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
import hashlib

# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Explainable Brain Tumor Detection",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# 2. PROJECT PATHS & MODEL CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

MODEL_PATH = (
    PROJECT_ROOT
    / "results"
    / "transfer_models"
    / "best_vit_v2_cls.weights.h5"
)

IMG_SIZE = 128
NUM_CLASSES = 3

CLASS_NAMES = [
    "Glioma",
    "Meningioma",
    "Pituitary"
]

PATCH_SIZE = 16
NUM_PATCHES = 64
EMBED_DIM = 128
NUM_TRANSFORMER_BLOCKS = 4
NUM_HEADS = 4
MLP_DIM = 256
DROPOUT_RATE = 0.1

# ============================================================
# 3. PROFESSIONAL UI STYLING
# ============================================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: #0b1120;
    color: #e2e8f0;
}

[data-testid="stSidebar"] {
    background: #111c30;
    border-right: 1px solid #26354d;
}

[data-testid="stHeader"] {
    background: transparent;
}

.hero {
    background: linear-gradient(120deg, #13243d, #172e4c);
    border: 1px solid #294563;
    border-radius: 18px;
    padding: 32px;
    margin-bottom: 25px;
}

.hero-label {
    color: #38bdf8;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
}

.hero-title {
    color: #f8fafc;
    font-size: 34px;
    font-weight: 800;
    margin-top: 10px;
    margin-bottom: 10px;
}

.hero-description {
    color: #a8bad0;
    font-size: 14px;
    line-height: 1.8;
    max-width: 850px;
}

.section-heading {
    font-size: 21px;
    font-weight: 700;
    color: #f1f5f9;
    margin-top: 15px;
    margin-bottom: 6px;
}

.section-caption {
    font-size: 13px;
    color: #94a3b8;
    margin-bottom: 20px;
}

.info-card {
    background: #142238;
    border: 1px solid #263c59;
    border-radius: 14px;
    padding: 20px;
    min-height: 125px;
}

.card-label {
    color: #94a3b8;
    font-size: 12px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1px;
}

.card-value {
    color: #f8fafc;
    font-size: 25px;
    font-weight: 800;
    margin-top: 10px;
}

.card-subtitle {
    color: #94a3b8;
    font-size: 12px;
    margin-top: 5px;
    line-height: 1.6;
}

.result-card {
    background: linear-gradient(135deg, #142d46, #173b54);
    border: 1px solid #28749a;
    border-radius: 16px;
    padding: 24px;
    min-height: 145px;
}

.result-label {
    color: #7dd3fc;
    font-size: 12px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1.5px;
}

.result-value {
    color: #ffffff;
    font-size: 30px;
    font-weight: 800;
    margin-top: 8px;
}

.result-description {
    color: #b6c9dc;
    font-size: 13px;
    margin-top: 8px;
}

.explanation-box {
    background: #142238;
    border-left: 4px solid #38bdf8;
    padding: 17px 20px;
    border-radius: 8px;
    color: #cbd5e1;
    font-size: 13px;
    line-height: 1.8;
    margin: 12px 0 20px 0;
}

.step-card {
    background: #142238;
    border: 1px solid #263c59;
    border-radius: 12px;
    padding: 20px;
    min-height: 150px;
    margin-bottom: 12px;
}

.step-number {
    color: #38bdf8;
    font-size: 12px;
    font-weight: 800;
    letter-spacing: 1px;
}

.step-title {
    color: #f1f5f9;
    font-size: 16px;
    font-weight: 700;
    margin-top: 12px;
}

.step-description {
    color: #94a3b8;
    font-size: 12px;
    line-height: 1.7;
    margin-top: 8px;
}

.disclaimer {
    background: #29251b;
    border: 1px solid #66502a;
    border-radius: 10px;
    padding: 14px 17px;
    color: #e8d5a7;
    font-size: 12px;
    line-height: 1.7;
    margin-top: 20px;
}

div.stButton > button {
    background: #0284c7;
    color: white;
    border: none;
    border-radius: 9px;
    font-weight: 700;
    min-height: 44px;
}

div.stButton > button:hover {
    background: #0369a1;
    color: white;
    border: none;
}

div[data-testid="stMetric"] {
    background: #142238;
    border: 1px solid #263c59;
    border-radius: 12px;
    padding: 16px;
}

div[data-testid="stMetricLabel"] {
    color: #94a3b8;
}

div[data-testid="stMetricValue"] {
    color: #f8fafc;
}

.stTabs [data-baseweb="tab"] {
    color: #94a3b8;
    font-weight: 600;
}

.stTabs [aria-selected="true"] {
    color: #38bdf8 !important;
}

hr {
    border-color: #26354d;
}
.hero {
    background: linear-gradient(120deg, #13243d, #172e4c);
    border: 1px solid #294563;
    border-radius: 18px;
    padding: 32px;
    margin-bottom: 25px;
}

.hero-label {
    color: #38bdf8;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
}

.hero-title {
    color: #f8fafc;
    font-size: 34px;
    font-weight: 800;
    margin-top: 10px;
    margin-bottom: 10px;
}

.hero-description {
    color: #a8bad0;
    font-size: 14px;
    line-height: 1.8;
    max-width: 850px;
}
</style>
""", unsafe_allow_html=True)


# ============================================================
# 4. VISION TRANSFORMER COMPONENTS
# ============================================================

class PatchEmbedding(layers.Layer):

    def __init__(self, embed_dim, patch_size, **kwargs):
        super().__init__(**kwargs)

        self.embed_dim = embed_dim
        self.patch_size = patch_size

        self.projection = layers.Conv2D(
            filters=embed_dim,
            kernel_size=patch_size,
            strides=patch_size,
            padding="valid"
        )

    def call(self, images):

        x = self.projection(images)
        batch_size = tf.shape(x)[0]

        return tf.reshape(
            x,
            (batch_size, -1, self.embed_dim)
        )


class PositionalEmbedding(layers.Layer):

    def __init__(self, num_patches, embed_dim, **kwargs):
        super().__init__(**kwargs)

        self.num_patches = num_patches
        self.embed_dim = embed_dim

        self.position_embedding = layers.Embedding(
            input_dim=num_patches,
            output_dim=embed_dim
        )

    def call(self, inputs):

        positions = tf.range(
            start=0,
            limit=self.num_patches,
            delta=1
        )

        positions = self.position_embedding(positions)

        return inputs + positions


class MultiHeadSelfAttention(layers.Layer):

    def __init__(self, embed_dim, num_heads, **kwargs):
        super().__init__(**kwargs)

        self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.attention = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embed_dim // num_heads,
            dropout=0.0
        )

        self.last_attention_scores = None

    def call(self, inputs, training=False):

        output, attention_scores = self.attention(
            query=inputs,
            value=inputs,
            key=inputs,
            training=training,
            return_attention_scores=True
        )

        self.last_attention_scores = attention_scores

        return output


class TransformerEncoder(layers.Layer):

    def __init__(
        self,
        embed_dim,
        num_heads,
        mlp_dim,
        dropout_rate,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.norm1 = layers.LayerNormalization(epsilon=1e-6)

        self.attention = MultiHeadSelfAttention(
            embed_dim=embed_dim,
            num_heads=num_heads
        )

        self.dropout1 = layers.Dropout(dropout_rate)

        self.norm2 = layers.LayerNormalization(epsilon=1e-6)

        self.mlp = tf.keras.Sequential([
            layers.Dense(
                mlp_dim,
                activation=tf.keras.activations.gelu
            ),
            layers.Dropout(dropout_rate),
            layers.Dense(embed_dim),
            layers.Dropout(dropout_rate)
        ])

    def call(self, inputs, training=False):

        x = self.norm1(inputs)

        x = self.attention(x, training=training)

        x = self.dropout1(x, training=training)

        x = x + inputs

        y = self.norm2(x)

        y = self.mlp(y, training=training)

        return x + y


class ViTWithCLS(tf.keras.Model):

    def __init__(
        self,
        img_size,
        patch_size,
        embed_dim,
        num_heads,
        num_blocks,
        mlp_dim,
        num_classes,
        dropout_rate,
        **kwargs
    ):

        super().__init__(**kwargs)

        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_blocks = num_blocks
        self.mlp_dim = mlp_dim
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate

        self.num_patches = (img_size // patch_size) ** 2

        self.patch_embedding = PatchEmbedding(
            embed_dim=embed_dim,
            patch_size=patch_size
        )

        self.cls_token = self.add_weight(
            name="cls_token",
            shape=(1, 1, embed_dim),
            initializer="random_normal",
            trainable=True
        )

        self.position_embedding = layers.Embedding(
            input_dim=self.num_patches + 1,
            output_dim=embed_dim
        )

        self.transformer_blocks = [
            TransformerEncoder(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_dim=mlp_dim,
                dropout_rate=dropout_rate
            )
            for _ in range(num_blocks)
        ]

        self.norm = layers.LayerNormalization(epsilon=1e-6)

        self.classifier = tf.keras.Sequential([
            layers.Dense(
                128,
                activation=tf.keras.activations.gelu
            ),
            layers.Dropout(dropout_rate),
            layers.Dense(
                num_classes,
                activation="softmax"
            )
        ])

    def call(self, inputs, training=False):

        x = self.patch_embedding(inputs)

        batch_size = tf.shape(x)[0]

        cls_tokens = tf.broadcast_to(
            self.cls_token,
            (batch_size, 1, self.embed_dim)
        )

        x = tf.concat([cls_tokens, x], axis=1)

        positions = tf.range(
            start=0,
            limit=self.num_patches + 1,
            delta=1
        )

        x = x + self.position_embedding(positions)

        for block in self.transformer_blocks:
            x = block(x, training=training)

        x = self.norm(x)

        cls_output = x[:, 0]

        return self.classifier(
            cls_output,
            training=training
        )


# ============================================================
# 5. LOAD TRAINED MODEL
# ============================================================

@st.cache_resource
def load_vit_model():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model weights not found: {MODEL_PATH}"
        )

    model = ViTWithCLS(
        img_size=IMG_SIZE,
        patch_size=PATCH_SIZE,
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        num_blocks=NUM_TRANSFORMER_BLOCKS,
        mlp_dim=MLP_DIM,
        num_classes=NUM_CLASSES,
        dropout_rate=DROPOUT_RATE
    )

    dummy_input = tf.zeros(
        (1, IMG_SIZE, IMG_SIZE, 1),
        dtype=tf.float32
    )

    model(dummy_input, training=False)

    model.load_weights(MODEL_PATH)

    return model


# ============================================================
# 6. IMAGE PREPROCESSING
# ============================================================

def preprocess_mri(image):

    if len(image.shape) == 3:
        image = cv2.cvtColor(
            image,
            cv2.COLOR_RGB2GRAY
        )

    image = cv2.resize(
        image,
        (IMG_SIZE, IMG_SIZE),
        interpolation=cv2.INTER_AREA
    )

    image = image.astype(np.float32) / 255.0

    image = image[..., np.newaxis]

    image = image[np.newaxis, ...]

    return image


# ============================================================
# 7. NATIVE ViT ATTENTION EXTRACTION
# ============================================================

def extract_attention_map(model, model_input):

    predictions = model(
        model_input,
        training=False
    )

    final_block = model.transformer_blocks[-1]

    attention_scores = (
        final_block.attention.last_attention_scores
    )

    if attention_scores is None:
        raise RuntimeError(
            "Attention scores were not captured."
        )

    attention = attention_scores.numpy()[0]

    # CLS token to image patches
    cls_to_patches = attention[:, 0, 1:]

    # Average across attention heads
    attention_map = np.mean(
        cls_to_patches,
        axis=0
    )

    grid_size = int(np.sqrt(len(attention_map)))

    attention_map = attention_map.reshape(
        grid_size,
        grid_size
    )

    # Normalize attention values
    attention_map = (
        attention_map - attention_map.min()
    )

    attention_map = attention_map / (
        attention_map.max() + 1e-8
    )

    # Resize to MRI resolution
    attention_map = cv2.resize(
        attention_map.astype(np.float32),
        (IMG_SIZE, IMG_SIZE),
        interpolation=cv2.INTER_LINEAR
    )

    attention_map = (
        attention_map - attention_map.min()
    )

    attention_map = attention_map / (
        attention_map.max() + 1e-8
    )

    return predictions.numpy()[0], attention_map


# ============================================================
# 8. HELPER FUNCTIONS
# ============================================================

def show_info_card(label, value, subtitle):

    st.markdown(
        f"""
        <div class="info-card">
            <div class="card-label">{label}</div>
            <div class="card-value">{value}</div>
            <div class="card-subtitle">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def create_attention_figure(
    grayscale_image,
    attention_map,
    selected_cmap,
    alpha_val
):

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(14, 4.5),
        facecolor="#142238"
    )

    titles = [
        "Original MRI",
        "ViT CLS Attention Map",
        "MRI + Attention Overlay"
    ]

    for ax in axes:
        ax.set_facecolor("#142238")
        ax.axis("off")

    axes[0].imshow(
        grayscale_image,
        cmap="gray"
    )

    axes[1].imshow(
        attention_map,
        cmap=selected_cmap,
        vmin=0,
        vmax=1
    )

    axes[2].imshow(
        grayscale_image,
        cmap="gray"
    )

    axes[2].imshow(
        attention_map,
        cmap=selected_cmap,
        alpha=alpha_val,
        vmin=0,
        vmax=1
    )

    for ax, title in zip(axes, titles):

        ax.set_title(
            title,
            color="#f1f5f9",
            fontsize=12,
            fontweight="bold",
            pad=12
        )

    plt.tight_layout()

    return fig


# ============================================================
# 9. SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("## 🧠 Explainable Brain Tumor Detection")

    st.caption("AI-powered MRI Classification & Visualization")

    st.divider()

    st.markdown("### Model Information")

    st.write("**Architecture:** Vision Transformer")
    st.write("**Attention:** Multi-Head Self-Attention")
    st.write("**Input:** 128 × 128 Grayscale MRI")
    st.write("**Patch Size:** 16 × 16")
    st.write("**Transformer Blocks:** 4")
    st.write("**Attention Heads:** 4")
    st.write("**Classes:** 3")

    st.divider()

    st.markdown("### Visualization Settings")

    selected_cmap = st.selectbox(
        "Attention Colormap",
        ["turbo", "jet", "plasma", "viridis", "hot"],
        index=0
    )

    alpha_val = st.slider(
        "Overlay Transparency",
        min_value=0.1,
        max_value=0.9,
        value=0.45,
        step=0.05
    )

    st.divider()

    st.caption("Weights: best_vit_v2_cls.weights.h5")
    st.caption("Academic Research Demonstration")



# ============================================================
# 10. MAIN HEADER
# ============================================================

st.markdown(
    """
    <style>
    .main-title {
        font-size: 34px;
        font-weight: 800;
        color: #f8fafc;
        margin-bottom: 10px;
    }
    .main-description {
        color: #a8bad0;
        font-size: 14px;
        line-height: 1.8;
        max-width: 850px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    '<div class="main-title">Explainable Brain Tumor Detection</div>',
    unsafe_allow_html=True
)

st.markdown(
    "An explainable brain tumor classification system powered by "
    "Vision Transformers. Analyze MRI images, identify predicted "
    "tumor categories, and visualize native self-attention patterns "
    "used by the model."
)

# ============================================================
# 11. MAIN NAVIGATION
# ============================================================

tab_overview, tab_analysis, tab_methodology = st.tabs([
    "Project Overview",
    "MRI Analysis",
    "Model Methodology"
])


# ============================================================
# TAB 1: PROJECT OVERVIEW
# ============================================================

with tab_overview:

    st.markdown(
        '<div class="section-heading">System Overview</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-caption">'
        'An overview of the model architecture and system capabilities.'
        '</div>',
        unsafe_allow_html=True
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        show_info_card(
            "AI Architecture",
            "ViT",
            "Vision Transformer"
        )

    with c2:
        show_info_card(
            "Input Resolution",
            "128 × 128",
            "Grayscale MRI"
        )

    with c3:
        show_info_card(
            "Attention Heads",
            "04",
            "Multi-Head Attention"
        )

    with c4:
        show_info_card(
            "Classification",
            "03",
            "Tumor Categories"
        )

    st.markdown(
        '<div class="section-heading">Tumor Categories</div>',
        unsafe_allow_html=True
    )

    tumor_cols = st.columns(3)

    tumor_info = [
        (
            "Glioma",
            "A tumor category originating from glial cells."
        ),
        (
            "Meningioma",
            "A tumor category associated with the meninges."
        ),
        (
            "Pituitary",
            "A tumor category associated with the pituitary gland."
        )
    ]

    for col, (name, description) in zip(
        tumor_cols,
        tumor_info
    ):

        with col:

            st.markdown(
                f"""
                <div class="info-card">
                    <div class="card-value"
                         style="font-size:20px;">
                        {name}
                    </div>
                    <div class="card-subtitle"
                         style="line-height:1.7;">
                        {description}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

    st.markdown(
        '<div class="section-heading">How the System Works</div>',
        unsafe_allow_html=True
    )

    steps = [
        (
            "01",
            "MRI Input",
            "Upload a brain MRI image in JPG or PNG format."
        ),
        (
            "02",
            "Preprocessing",
            "Convert to grayscale, resize to 128 × 128 and normalize."
        ),
        (
            "03",
            "ViT Processing",
            "Extract image patches and process them through transformer blocks."
        ),
        (
            "04",
            "Classification",
            "Use the CLS token representation to predict a tumor category."
        ),
        (
            "05",
            "Explanation",
            "Extract final-block CLS-to-patch attention for visualization."
        )
    ]

    step_cols = st.columns(5)

    for col, (number, title, description) in zip(
        step_cols,
        steps
    ):

        with col:

            st.markdown(
                f"""
                <div class="step-card">
                    <div class="step-number">STEP {number}</div>
                    <div class="step-title">{title}</div>
                    <div class="step-description">{description}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

    st.markdown("""
    <div class="disclaimer">
        <strong>Research Disclaimer:</strong>
        This application is an academic AI research demonstration.
        Predictions and attention visualizations are not a substitute
        for professional medical diagnosis.
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# TAB 2: MRI ANALYSIS - UPLOAD ONLY
# ============================================================

with tab_analysis:

    st.markdown(
        '<div class="section-heading">MRI Image Analysis</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-caption">'
        'Upload a brain MRI image to run the trained Vision Transformer.'
        '</div>',
        unsafe_allow_html=True
    )

    uploaded_file = st.file_uploader(
        "Upload MRI Image",
        type=["jpg", "jpeg", "png"],
        help="Supported formats: JPG, JPEG and PNG",
        key="mri_uploader"
    )

    active_image = None
    active_name = ""

    if uploaded_file is not None:

        active_name = uploaded_file.name

        image_bytes = uploaded_file.getvalue()

        # Identify the actual uploaded file, not just its filename
        file_hash = hashlib.md5(image_bytes).hexdigest()

        if st.session_state.get("last_uploaded_hash") != file_hash:

            st.session_state.pop("analysis_result", None)

            st.session_state["last_uploaded_hash"] = file_hash

        try:

            active_image = Image.open(
                uploaded_file
            ).convert("RGB")

        except Exception:

            st.error("Unable to read the uploaded image. Please upload a valid JPG or PNG image.")

    if active_image is not None:

        st.markdown("---")

        preview_col, details_col = st.columns([1, 1.4])

        with preview_col:

            st.markdown("#### Uploaded MRI Preview")

            st.image(
                active_image,
                caption=active_name,
                use_container_width=True
            )

        with details_col:

            st.markdown("#### Image Information")

            st.write(f"**Filename:** {active_name}")

            st.write(
                f"**Original dimensions:** "
                f"{active_image.size[0]} × {active_image.size[1]}"
            )

            st.write("**Model input:** 128 × 128")

            st.write("**Image channels:** 1 (Grayscale)")

            st.write("**Preprocessing:** Resize and normalization")

        run_analysis = st.button(
            "Run MRI Analysis",
            use_container_width=True
        )

        if run_analysis:

            with st.spinner(
                "Loading trained Vision Transformer..."
            ):

                try:

                    model = load_vit_model()

                except Exception as e:

                    st.error(
                        f"Model loading failed: {e}"
                    )

                    st.stop()

            original_image = np.array(active_image)

            model_input = preprocess_mri(
                original_image
            )

            with st.spinner(
                "Analyzing MRI and extracting attention maps..."
            ):

                try:

                    probabilities, attention_map = (
                        extract_attention_map(
                            model,
                            model_input
                        )
                    )

                except Exception as e:

                    st.error(
                        f"Analysis failed: {e}"
                    )

                    st.stop()

            predicted_index = int(
                np.argmax(probabilities)
            )

            predicted_class = CLASS_NAMES[
                predicted_index
            ]

            confidence = float(
                probabilities[predicted_index]
            )

            st.session_state["analysis_result"] = {
                "probabilities": probabilities,
                "attention_map": attention_map,
                "predicted_class": predicted_class,
                "confidence": confidence,
                "original_image": original_image,
                "active_name": active_name,
                "file_hash": file_hash
            }

    # Display results only for the currently uploaded image

    result = st.session_state.get("analysis_result")

    if (
        result is not None
        and uploaded_file is not None
        and result.get("file_hash") == st.session_state.get("last_uploaded_hash")
    ):

        probabilities = result["probabilities"]
        attention_map = result["attention_map"]
        predicted_class = result["predicted_class"]
        confidence = result["confidence"]
        original_image = result["original_image"]

        st.divider()

        st.markdown(
            '<div class="section-heading">Analysis Results</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="section-caption">'
            'Classification output generated by the trained ViT model.'
            '</div>',
            unsafe_allow_html=True
        )

        result_col, confidence_col = st.columns([1.3, 1])

        with result_col:

            st.markdown(
                f"""
                <div class="result-card">
                    <div class="result-label">
                        Predicted Category
                    </div>
                    <div class="result-value">
                        {predicted_class}
                    </div>
                    <div class="result-description">
                        Highest probability among the three model classes
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with confidence_col:

            st.metric(
                "Prediction Confidence",
                f"{confidence:.2%}"
            )

            st.progress(confidence)

            st.caption(
                "Softmax probability for the predicted class."
            )

        # Class-wise probabilities

        st.markdown(
            '<div class="section-heading">'
            'Class-wise Probability Distribution'
            '</div>',
            unsafe_allow_html=True
        )

        probability_cols = st.columns(3)

        for i, col in enumerate(probability_cols):

            with col:

                st.metric(
                    CLASS_NAMES[i],
                    f"{float(probabilities[i]):.2%}"
                )

                st.progress(
                    float(probabilities[i])
                )

        # Attention visualization

        st.divider()

        st.markdown(
            '<div class="section-heading">Visual Explanation</div>',
            unsafe_allow_html=True
        )

        st.markdown("""
        <div class="explanation-box">
            <strong>Understanding the attention map:</strong>
            This visualization represents CLS-to-patch attention
            extracted from the final Transformer block and averaged
            across four attention heads. Brighter regions indicate
            relatively higher attention weights. This is an attention
            visualization, not a verified tumor segmentation mask.
        </div>
        """, unsafe_allow_html=True)

        grayscale_image = cv2.cvtColor(
            original_image,
            cv2.COLOR_RGB2GRAY
        )

        grayscale_image = cv2.resize(
            grayscale_image,
            (IMG_SIZE, IMG_SIZE),
            interpolation=cv2.INTER_AREA
        )

        fig = create_attention_figure(
            grayscale_image,
            attention_map,
            selected_cmap,
            alpha_val
        )

        st.pyplot(
            fig,
            use_container_width=True
        )

        plt.close(fig)

        # Download attention map

        attention_uint8 = (
            attention_map * 255
        ).astype(np.uint8)

        _, encoded_attention = cv2.imencode(
            ".png",
            attention_uint8
        )

        st.download_button(
            "Download Attention Map",
            data=encoded_attention.tobytes(),
            file_name="vit_attention_map.png",
            mime="image/png"
        )

        # Technical details

        with st.expander("Technical Analysis Details"):

            st.write(
                f"**Input resolution:** {IMG_SIZE} × {IMG_SIZE}"
            )

            st.write(
                f"**Patch size:** {PATCH_SIZE} × {PATCH_SIZE}"
            )

            st.write(
                f"**Number of image patches:** {NUM_PATCHES}"
            )

            st.write(
                f"**Transformer blocks:** {NUM_TRANSFORMER_BLOCKS}"
            )

            st.write(
                f"**Attention heads:** {NUM_HEADS}"
            )

            st.write(
                "**Attention source:** Final block, CLS-to-patch attention"
            )

            st.write(
                "**Model weights:** best_vit_v2_cls.weights.h5"
            )

        st.markdown("""
        <div class="disclaimer">
            <strong>Important:</strong>
            Model confidence is not equivalent to diagnostic certainty.
            Attention weights indicate where the model distributes
            attention, but do not establish the presence or precise
            boundaries of a tumor.
        </div>
        """, unsafe_allow_html=True)

    elif uploaded_file is None:

        st.info(
            "Upload an MRI image to begin the analysis."
        )


# ============================================================
# TAB 3: MODEL METHODOLOGY
# ============================================================

with tab_methodology:

    st.markdown(
        '<div class="section-heading">Vision Transformer Architecture</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-caption">'
        'Technical overview of the model used for brain MRI classification.'
        '</div>',
        unsafe_allow_html=True
    )

    architecture_steps = [
        (
            "01",
            "Input MRI",
            "128 × 128 grayscale image"
        ),
        (
            "02",
            "Patch Embedding",
            "16 × 16 patches projected into 128-dimensional vectors"
        ),
        (
            "03",
            "CLS + Position Embedding",
            "Learnable CLS token and positional information"
        ),
        (
            "04",
            "Transformer Encoder",
            "Four encoder blocks with four attention heads"
        ),
        (
            "05",
            "Classification Head",
            "CLS representation passed through dense layers"
        ),
        (
            "06",
            "Output",
            "Softmax probabilities for three tumor categories"
        )
    ]

    for i in range(0, len(architecture_steps), 2):

        cols = st.columns(2)

        for col, step in zip(
            cols,
            architecture_steps[i:i + 2]
        ):

            number, title, description = step

            with col:

                st.markdown(
                    f"""
                    <div class="step-card">
                        <div class="step-number">
                            STAGE {number}
                        </div>
                        <div class="step-title">
                            {title}
                        </div>
                        <div class="step-description">
                            {description}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

    st.divider()

    st.markdown("#### Model Specifications")

    specs = {
        "Architecture": "Vision Transformer with CLS token",
        "Image Input": "128 × 128 × 1",
        "Patch Dimensions": "16 × 16",
        "Number of Patches": "64",
        "Embedding Dimension": "128",
        "Transformer Blocks": "4",
        "Attention Heads": "4",
        "MLP Dimension": "256",
        "Dropout": "0.1",
        "Output Activation": "Softmax",
        "Number of Classes": "3",
        "Attention Visualization": "Final-block CLS-to-patch attention"
    }

    for key, value in specs.items():

        left, right = st.columns([1, 2])

        with left:
            st.write(f"**{key}**")

        with right:
            st.write(value)

    st.divider()

    st.markdown("#### Attention Extraction Method")

    st.code("""
Final Transformer Block
          |
          v
Multi-Head Attention Scores
          |
          v
CLS Token Attention
          |
          v
Extract CLS-to-Patch Weights
          |
          v
Average Across 4 Heads
          |
          v
Reshape into 8 x 8 Grid
          |
          v
Resize to 128 x 128
          |
          v
Attention Heatmap
    """, language="text")


# ============================================================
# 12. FOOTER
# ============================================================

st.divider()

footer_col1, footer_col2 = st.columns([2, 1])

with footer_col1:

    st.caption(
        "Explainable Brain Tumor Detection | "
        "Vision Transformer | Native Self-Attention"
    )

with footer_col2:

    st.caption("Academic Research Demonstration")
