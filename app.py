import os
import re
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from sentence_transformers import SentenceTransformer
import lightgbm as lgb

st.set_page_config(
    page_title="Multimodal Pricing Engine",
    page_icon="🏷️",
    layout="wide"
)

st.markdown("""
<style>
    .metric-card {
        background-color: #111827;
        border: 1px solid #374151;
        border-radius: 12px;
        padding: 24px;
        text-align: center;
        margin-top: 10px;
    }
    .price-value {
        font-size: 3rem;
        font-weight: 800;
        color: #10B981;
        margin: 6px 0;
    }
    .tag {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.85rem;
        background: #1F2937;
        color: #D1D5DB;
        border: 1px solid #374151;
        margin: 2px;
    }
</style>
""", unsafe_allow_html=True)

# 1. Load Models & Checkpoint
@st.cache_resource
def load_pipeline():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Text Backbone (384-dim)
    text_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)

    # Vision Backbone (512-dim)
    img_model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
    img_model.fc = nn.Identity()
    img_model = img_model.to(device)
    img_model.eval()

    # Trained 906-feature LightGBM
    model_path = "models/pricing_lgbm.txt"
    if not os.path.exists(model_path):
        st.error(f"Missing {model_path}. Run `python train_lgbm.py` first.")
        st.stop()
    booster = lgb.Booster(model_file=model_path)

    return device, text_model, img_model, booster

device, text_model, img_model, booster = load_pipeline()

# 2. Image Transform
img_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def extract_tabular_signals(text: str):
    t = text.lower()

    # 1. Pack Quantity
    pack_match = re.search(r'(?:pack|count|set|box|lot)\s*(?:of)?\s*[:\-]?\s*(\d+)', t)
    pack_qty = int(pack_match.group(1)) if pack_match and 1 <= int(pack_match.group(1)) <= 500 else 1

    # 2. Weight in Grams
    wt_match = re.search(r'(\d+(?:\.\d+)?)\s*(kg|kilogram|g|gram|gm|oz|pound|lb)\b', t)
    weight_g = np.nan
    if wt_match:
        val, unit = float(wt_match.group(1)), wt_match.group(2)
        weight_g = val * 1000.0 if "kg" in unit else val

    # 3. Text Statistics
    char_count = len(text)
    word_count = len(text.split())
    num_digits = sum(c.isdigit() for c in text)
    caps_ratio = sum(c.isupper() for c in text) / (len(text) + 1)

    # 4. Catalog Flags
    is_refurbished = int(bool(re.search(r'\b(refurbished|renewed|used)\b', t)))
    is_premium = int(bool(re.search(r'\b(pro|max|ultra|plus|edition)\b', t)))
    has_warranty = int(bool(re.search(r'\bwarranty\b', t)))

    # Exact 10 tabular features matching your trained model
    feats = np.array([
        pack_qty, weight_g, np.nan,
        char_count, word_count, num_digits, caps_ratio,
        is_refurbished, is_premium, has_warranty
    ], dtype=np.float32).reshape(1, -1)

    # Return all 4 values expected by line 151
    return feats, pack_qty, weight_g, is_premium

# 4. Streamlit UI
st.title("🏷️ Dynamic Pricing Engine")
st.caption(f"Hardware: `{device.upper()}` | Input Shape: `906 Features (10 Tabular + 384 Text + 512 Vision)`")

col1, col2 = st.columns([1.1, 1], gap="large")

with col1:
    st.subheader("1. Product Details")
    product_text = st.text_area(
        "Product Title & Catalog Description",
        placeholder="Paste full product title and description here...",
        height=180
    )
    uploaded_image = st.file_uploader("Upload Product Image", type=["jpg", "jpeg", "png"])
    if uploaded_image:
        st.image(uploaded_image, caption="Product Photo", width=260)

with col2:
    st.subheader("2. Price Evaluation")
    if st.button("Estimate Optimal Price", type="primary", use_container_width=True):
        if not product_text.strip():
            st.error("Please enter a product title or description.")
        else:
            with st.spinner("Extracting multimodal representations..."):
                # Tabular (1, 10)
                tab_feats, p_qty, p_wt, is_prem = extract_tabular_signals(product_text)

                # Text (1, 384)
                text_emb = text_model.encode([product_text], normalize_embeddings=True, convert_to_numpy=True)

                # Vision (1, 512)
                if uploaded_image:
                    img = Image.open(uploaded_image).convert("RGB")
                    tensor = img_transform(img).unsqueeze(0).to(device)
                    with torch.no_grad():
                        img_emb = img_model(tensor).cpu().numpy()
                else:
                    img_emb = np.zeros((1, 512), dtype=np.float32)

                # Exact 906 features
                X_full = np.hstack([tab_feats, text_emb, img_emb])
                pred_log = booster.predict(X_full)
                price = float(np.expm1(pred_log)[0])

            st.markdown(f"""
            <div class="metric-card">
                <span style="color: #9CA3AF; font-size: 0.9rem; letter-spacing: 1px;">PREDICTED MARKET PRICE</span>
                <div class="price-value">${price:,.2f}</div>
                <div style="margin-top: 10px;">
                    <span class="tag">Pack: {p_qty}</span>
                    <span class="tag">Weight: {'N/A' if np.isnan(p_wt) else f'{p_wt:.0f}g'}</span>
                    <span class="tag">Tier: {'Premium' if is_prem else 'Standard'}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)