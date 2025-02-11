import streamlit as st
import torch
import pandas as pd
import numpy as np
from PIL import Image
import hydra
from omegaconf import DictConfig, OmegaConf
import os
from os.path import join
from models.Tip_utils.Tip_downstream import TIPBackbone
import base64

def get_image_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

background_image = get_image_base64(
    "./streamlit/parking_car_2.jpg"
)
tilda_base64 = get_image_base64(
    "./streamlit/tilda_web_logo.png"
)
boda_base64 = get_image_base64(
    "./streamlit/BODA_LOGO.png"
)

car_image = get_image_base64(
    "./streamlit/car_image_without_background.png"
)

st.set_page_config(
    page_title="자동차 판매량 예측 서비스",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    f"""
    <style>
    html {{
        scroll-behavior: smooth;
    }}

    body {{
        scroll-snap-type: y mandatory;
        overflow-y: scroll;
    }}

    .logo-container, .main-container {{
        scroll-snap-align: start;
        scroll-snap-stop: always;
    }}

    .stApp {{
        background-image: url(data:image/jpeg;base64,{background_image});
        background-attachment: fixed;
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        min-height: 100vh;
        position: relative;
    }}

    .stApp::before {{
        content: "";
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: rgba(0, 0, 0, 0.7);
        z-index: 0;
    }}

    .logo-container {{
        display: flex;
        justify-content: center;
        align-items: center;
        gap: 20px;
        padding: 0;
        margin: 0;
        position: relative;
        z-index: 1;
        background: linear-gradient(
            180deg,
            rgba(255, 255, 255, 0) 0%,
            rgba(255, 255, 255, 0.3) 20%,
            rgba(255, 255, 255, 0.6) 40%,
            rgba(255, 255, 255, 0.6) 60%,
            rgba(255, 255, 255, 0.3) 80%,
            rgba(255, 255, 255, 0) 100%
        );
        width: 100vw;
        height: 80vh;
        margin-left: calc(-50vw + 50%);
        margin-right: calc(-50vw + 50%);
    }}
    
    .logo-image.tilda {{
        height: 200px;
        width: auto;
        object-fit: contain;
        transition: transform 0.3s ease;
    }}
    
    .logo-image.boda {{
        height: 600px;
        width: auto;
        object-fit: contain;
        transition: transform 0.3s ease;
        margin-top: 20px;
        margin-left: -80px;
    }}
    
    .logo-image:hover {{
        transform: scale(1.1);
    }}
    
    .cross {{
        font-size: 24px;
        color: #333;
        margin: 0 20px;
        font-weight: bold;
    }}

    .main-container {{
        display: flex;
        justify-content: center;
        align-items: center;
        gap: 20px;
        padding: 0;
        margin: 0;
        position: relative;
        z-index: 1;
        width: 100vw;
        height: 70vh;
        margin-left: calc(-50vw + 50%);
        margin-right: calc(-50vw + 50%);
    }}

    .main-content {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        width: 80%;
        max-width: 1200px;
    }}

    .text-section {{
        flex: 1;
        padding-right: 50px;
    }}

    .main-heading {{
        font-size: 3.5rem;
        font-weight: 700;
        color: #ffffff;
        line-height: 1.2;
        margin-bottom: 30px;
    }}

    .image-section {{
        flex: 1;
        text-align: right;
    }}

    .car-image {{
        max-width: 600px;
        height: auto;
        filter: drop-shadow(5px 5px 15px rgba(0, 0, 0, 0.3));
    }}

    .predict-button {{
        display: flex;
        justify-content: center;
        align-items: center;
        background-color: rgba(41, 128, 185, 0.9);
        color: #ffffff !important;
        padding: 15px 40px;
        border-radius: 30px;
        text-decoration: none !important;
        font-size: 1.5rem;
        font-weight: 600;
        margin-top: 20px;
        transition: all 0.3s ease;
        border: 2px solid rgba(255, 255, 255, 0.2);
        backdrop-filter: blur(5px);
        width: 200px;
        height: 60px;
        white-space: nowrap;
        overflow: hidden;
    }}

    .predict-button:hover {{
        background-color: rgba(52, 152, 219, 0.9);
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(41, 128, 185, 0.3);
        border: 2px solid rgba(255, 255, 255, 0.4);
        color: #ffffff !important;
        text-decoration: none !important;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

@st.cache_resource
def load_model(checkpoint_path, config_path):
    try:
        with hydra.initialize(config_path="./configs"):
            cfg = hydra.compose(
                config_name="config_dvm_TIP", overrides=["dataset=dvm_all_server"]
            )

            cfg.data_base = "./data/base_features"
            cfg.field_lengths_tabular = join(
                cfg.data_base, "tabular_lengths_all_views_reordered.pt"
            )

            cfg.model = "resnet50"
            cfg.img_size = 128
            cfg.embedding_dim = 2048

        checkpoint = torch.load(checkpoint_path)

        model = TIPBackbone(cfg)
        model.eval()

        return model, cfg

    except Exception as e:
        st.error(f"모델 로드 중 오류 발생: {str(e)}")
        raise e


def main():
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url(data:image/jpeg;base64,{background_image});
            background-attachment: fixed;
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            min-height: 100vh;
            position: relative;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

    section1 = st.container()
    with section1:
        st.markdown(
            f"""
            <div class="logo-container">
                <img src="data:image/png;base64,{tilda_base64}" class="logo-image tilda" alt="Tilda Logo"/>
                <span class="cross">×</span>
                <img src="data:image/png;base64,{boda_base64}" class="logo-image boda" alt="BODA Logo"/>
            </div>
            """,
            unsafe_allow_html=True,
        )

    section2 = st.container()
    with section2:
        st.markdown(
            f"""
            <div class="main-container">
                <div class="main-content">
                    <div class="text-section">
                        <h1 class="main-heading">Predict the sales of new car!</h1>
                        <a href="/predict" class="predict-button">
                            Predict Now →
                        </a>
                    </div>
                    <div class="image-section">
                        <img src="data:image/png;base64,{car_image}" class="car-image" alt="Car Image"/>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

if __name__ == "__main__":
    main()
