# ============================================================
# EXPLAINABLE BRAIN TUMOR DETECTION
# Vision Transformer (ViT) with Native Multi-Head Self-Attention
# ============================================================

import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras import layers
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt

# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Explainable Brain Tumor Detection",
    page_icon="🧠",
    layout="wide"
)

# Custom CSS for clean medium visuals
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #38bdf8;
        margin-bottom: 0px;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #94a3b8;
        margin-bottom: 16px;
    }
    .metric-container {
        background: #1e293b;
        border-radius: 10px;
        padding: 16px;
        border: 1px solid rgba(255,255,255,0.08);
        text-align: center;
    }
    .expl-box {
        background: rgba(56, 189, 248, 0.08);
        border-left: 4px solid #38bdf8;
        padding: 12px 16px;
        border-radius: 0 8px 8px 0;
        color: #e2e8f0;
        font-size: 0.95rem;
        margin: 12px 0 20px 0;
    }
</style>
""", unsafe_allow_html=True)

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
# 3. VISION TRANSFORMER COMPONENTS
# ============================================================

class PatchEmbedding(layers.Layer):
    """Converts the MRI image into non-overlapping patches (16x16) and projects them."""
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
        return tf.reshape(x, (batch_size, -1, self.embed_dim))


class PositionalEmbedding(layers.Layer):
    """Adds learnable positional embeddings to patch tokens."""
    def __init__(self, num_patches, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.num_patches = num_patches
        self.embed_dim = embed_dim
        self.position_embedding = layers.Embedding(
            input_dim=num_patches,
            output_dim=embed_dim
        )

    def call(self, inputs):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        positions = self.position_embedding(positions)
        return inputs + positions


class MultiHeadSelfAttention(layers.Layer):
    """Multi-Head Self-Attention. Attention scores are stored for visualization."""
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
    """Standard Transformer encoder block with residual connections."""
    def __init__(self, embed_dim, num_heads, mlp_dim, dropout_rate, **kwargs):
        super().__init__(**kwargs)
        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.attention = MultiHeadSelfAttention(embed_dim=embed_dim, num_heads=num_heads)
        self.dropout1 = layers.Dropout(dropout_rate)
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)
        self.mlp = tf.keras.Sequential([
            layers.Dense(mlp_dim, activation=tf.keras.activations.gelu),
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
    """Vision Transformer with learnable CLS token."""
    def __init__(self, img_size, patch_size, embed_dim, num_heads, num_blocks, mlp_dim, num_classes, dropout_rate, **kwargs):
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
        self.patch_embedding = PatchEmbedding(embed_dim=embed_dim, patch_size=patch_size)

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
            layers.Dense(128, activation=tf.keras.activations.gelu),
            layers.Dropout(dropout_rate),
            layers.Dense(num_classes, activation="softmax")
        ])

    def call(self, inputs, training=False):
        x = self.patch_embedding(inputs)
        batch_size = tf.shape(x)[0]
        cls_tokens = tf.broadcast_to(self.cls_token, (batch_size, 1, self.embed_dim))
        x = tf.concat([cls_tokens, x], axis=1)

        positions = tf.range(start=0, limit=self.num_patches + 1, delta=1)
        x = x + self.position_embedding(positions)

        for block in self.transformer_blocks:
            x = block(x, training=training)

        x = self.norm(x)
        cls_output = x[:, 0]
        return self.classifier(cls_output, training=training)


# ============================================================
# 4. LOAD TRAINED MODEL
# ============================================================

@st.cache_resource
def load_vit_model():
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
    dummy_input = tf.zeros((1, IMG_SIZE, IMG_SIZE, 1), dtype=tf.float32)
    model(dummy_input, training=False)
    model.load_weights(MODEL_PATH)
    return model


# ============================================================
# 5. PREPROCESS MRI
# ============================================================

def preprocess_mri(image):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    image = image.astype(np.float32) / 255.0
    image = image[..., np.newaxis]
    image = image[np.newaxis, ...]
    return image


# ============================================================
# 6. EXTRACT NATIVE ViT ATTENTION
# ============================================================

def extract_attention_map(model, model_input):
    predictions = model(model_input, training=False)
    final_block = model.transformer_blocks[-1]
    attention_scores = final_block.attention.last_attention_scores
    if attention_scores is None:
        raise RuntimeError("Attention scores were not captured.")

    # Shape: (batch, heads, tokens, tokens)
    attention = attention_scores.numpy()[0]
    # CLS to 64 image patches across all 4 heads
    cls_to_patches = attention[:, 0, 1:]
    # Average across heads
    attention_map = np.mean(cls_to_patches, axis=0)

    grid_size = int(np.sqrt(len(attention_map)))  # 8
    attention_map = attention_map.reshape(grid_size, grid_size)

    # Normalize [0, 1]
    attention_map = attention_map - attention_map.min()
    attention_map = attention_map / (attention_map.max() + 1e-8)

    # Resize to original 128x128 MRI resolution
    attention_map = cv2.resize(
        attention_map.astype(np.float32),
        (IMG_SIZE, IMG_SIZE),
        interpolation=cv2.INTER_LINEAR
    )

    attention_map = attention_map - attention_map.min()
    attention_map = attention_map / (attention_map.max() + 1e-8)

    return predictions.numpy()[0], attention_map


# ============================================================
# 7. UI: TITLE & SIDEBAR
# ============================================================

st.markdown('<div class="main-title">🧠 Explainable Brain Tumor Detection</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Vision Transformer (ViT) with Native Multi-Head Self-Attention Visualization</div>', unsafe_allow_html=True)
st.divider()

with st.sidebar:
    st.header("Model Information")
    st.write("**Architecture:** Vision Transformer (ViT)")
    st.write("**Attention:** Multi-Head Self-Attention (4 Heads)")
    st.write("**Input:** 128 × 128 grayscale MRI")
    st.write("**Patch Size:** 16 × 16 (64 Patches)")
    st.write("**Classes:** 3")
    st.write("- Glioma")
    st.write("- Meningioma")
    st.write("- Pituitary")
    
    st.divider()
    st.markdown("#### Visual Settings")
    selected_cmap = st.selectbox("Heatmap Colormap", ["jet", "turbo", "plasma", "viridis", "hot"], index=0)
    alpha_val = st.slider("Overlay Transparency", 0.1, 0.9, 0.45, 0.05)


# ============================================================
# 8. FILE UPLOAD & SAMPLE SELECTION
# ============================================================

col_up, col_samples = st.columns([2, 1])

with col_up:
    uploaded_file = st.file_uploader(
        "Upload MRI Image (JPG / PNG)",
        type=["jpg", "jpeg", "png"]
    )

with col_samples:
    st.markdown("##### Or Select a Sample MRI:")
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("Glioma", use_container_width=True):
            st.session_state["sample_img"] = "samples/sample_glioma_01.jpg"
            st.rerun()
    with c2:
        if st.button("Meningioma", use_container_width=True):
            st.session_state["sample_img"] = "samples/sample_meningioma_01.jpg"
            st.rerun()
    with c3:
        if st.button("Pituitary", use_container_width=True):
            st.session_state["sample_img"] = "samples/sample_pituitary_01.jpg"
            st.rerun()

# Determine active image
active_image = None
active_name = ""

if uploaded_file is not None:
    active_image = Image.open(uploaded_file).convert("RGB")
    active_name = uploaded_file.name
elif "sample_img" in st.session_state and Path(st.session_state["sample_img"]).exists():
    active_image = Image.open(st.session_state["sample_img"]).convert("RGB")
    active_name = Path(st.session_state["sample_img"]).name
else:
    # Default to first sample if available
    default_p = "samples/sample_glioma_01.jpg"
    if Path(default_p).exists():
        active_image = Image.open(default_p).convert("RGB")
        active_name = "Sample Glioma MRI"


# ============================================================
# 9. RUN INFERENCE & DISPLAY RESULTS (MEDIUM VISUALS)
# ============================================================

if active_image is not None:
    original_image = np.array(active_image)

    # Load Model
    with st.spinner("Loading Vision Transformer..."):
        try:
            model = load_vit_model()
        except Exception as e:
            st.error(f"Error loading model: {e}")
            st.stop()

    # Preprocess
    model_input = preprocess_mri(original_image)

    # Predict & Extract Attention
    with st.spinner("Analyzing MRI with Self-Attention..."):
        try:
            probabilities, attention_map = extract_attention_map(model, model_input)
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.stop()

    predicted_index = int(np.argmax(probabilities))
    predicted_class = CLASS_NAMES[predicted_index]
    confidence = float(probabilities[predicted_index])

    st.divider()

    # --------------------------------------------------------
    # Prediction Header
    # --------------------------------------------------------
    st.subheader("Prediction Result")
    
    col_pred1, col_pred2 = st.columns(2)
    with col_pred1:
        st.metric("Predicted Tumor Class", predicted_class)
    with col_pred2:
        st.metric("Confidence", f"{confidence:.2%}")

    # --------------------------------------------------------
    # Class Probabilities Bar Chart
    # --------------------------------------------------------
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("Class Probabilities")
    
    probability_data = {
        CLASS_NAMES[i]: float(probabilities[i])
        for i in range(NUM_CLASSES)
    }
    st.bar_chart(probability_data)

    # --------------------------------------------------------
    # Native ViT Attention Explanation (Medium Visuals)
    # --------------------------------------------------------
    st.divider()
    st.subheader("Native ViT Attention Explanation")
    
    st.markdown("""
    <div class="expl-box">
        <strong>How to read this:</strong> The heatmap shows which spatial regions received the highest <strong>CLS-to-patch self-attention</strong> in the final Transformer block. 
        <strong>Red/Yellow areas</strong> highlight the exact tumor region the model focused on to make its prediction.
    </div>
    """, unsafe_allow_html=True)

    # Pre-render images
    grayscale_image = cv2.cvtColor(original_image, cv2.COLOR_RGB2GRAY) if len(original_image.shape) == 3 else original_image
    grayscale_image = cv2.resize(grayscale_image, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)

    # Medium-sized 3-column display: cleanly proportioned, easy to see and explain
    v1, v2, v3 = st.columns(3)

    with v1:
        fig1, ax1 = plt.subplots(figsize=(4.0, 4.0), facecolor='#0f172a', dpi=110)
        ax1.imshow(grayscale_image, cmap="gray")
        ax1.set_title("1. Original MRI", color="#e2e8f0", fontsize=11, pad=8)
        ax1.axis("off")
        plt.tight_layout()
        st.pyplot(fig1, use_container_width=True)
        plt.close(fig1)

    with v2:
        fig2, ax2 = plt.subplots(figsize=(4.0, 4.0), facecolor='#0f172a', dpi=110)
        ax2.imshow(attention_map, cmap=selected_cmap)
        ax2.set_title("2. ViT CLS Attention Map", color="#e2e8f0", fontsize=11, pad=8)
        ax2.axis("off")
        plt.tight_layout()
        st.pyplot(fig2, use_container_width=True)
        plt.close(fig2)

    with v3:
        fig3, ax3 = plt.subplots(figsize=(4.0, 4.0), facecolor='#0f172a', dpi=110)
        ax3.imshow(grayscale_image, cmap="gray")
        ax3.imshow(attention_map, cmap=selected_cmap, alpha=alpha_val)
        ax3.set_title("3. MRI + ViT Attention Overlay", color="#e2e8f0", fontsize=11, pad=8)
        ax3.axis("off")
        plt.tight_layout()
        st.pyplot(fig3, use_container_width=True)
        plt.close(fig3)

    # --------------------------------------------------------
    # Technical Details Expander
    # --------------------------------------------------------
    with st.expander("Technical Details"):
        st.write(f"**Input resolution:** {IMG_SIZE} × {IMG_SIZE}")
        st.write(f"**Patch size:** {PATCH_SIZE} × {PATCH_SIZE}")
        st.write(f"**Number of image patches:** {NUM_PATCHES} (8 × 8 grid)")
        st.write(f"**Transformer blocks:** {NUM_TRANSFORMER_BLOCKS}")
        st.write(f"**Attention heads per block:** {NUM_HEADS}")
        st.write("**Attention source:** Final Transformer block, CLS-to-patch attention")
        st.write(f"**Model Weights File:** `results/transfer_models/best_vit_v2_cls.weights.h5`")
