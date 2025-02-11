import streamlit as st
import base64
from PIL import Image
import sys
import os
import yaml
from typing import Dict
import torch
from inference import (
    load_model,
    load_config,
    infer_sales_amount_and_features,
    preprocess_data,
) 
import hydra
from omegaconf import DictConfig, OmegaConf
from models.Tip_utils.Tip_downstream import TIPBackbone

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

def get_image_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

background_image = get_image_base64(
    "./streamlit/parking_car_2.jpg"
)

def get_values_from_mapping(mapping: Dict) -> list:
    """config의 mapping에서 값 목록을 추출"""
    return [""] + list(mapping.values())

def predict_page():
    st.set_page_config(
        page_title="차량 판매량 예측",
        page_icon="🚗",
        layout="centered",
        initial_sidebar_state="collapsed" 
    )

    st.markdown(
        """
        <style>
        [data-testid="collapsedControl"] {
            display: none
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    CONFIG_PATH = "./configs/config.yaml"

    config = load_config(CONFIG_PATH)

    if "step" not in st.session_state:
        st.session_state.step = 1

    initial_states = {
        "genmodel": None,
        "entry_price": None,
        "maker": None,
        "color": None,
        "bodytype": None,
        "fuel_type": None,
        "gearbox": None,
        "first_release_year": None,
        "year": None,
        "wheelbase": None,
        "width": None,
        "length": None,
        "height": None,
        "seat_num": None,
        "door_num": None,
        "engine_size": None,
    }

    for key, value in initial_states.items():
        if key not in st.session_state:
            st.session_state[key] = value

    if "images" not in st.session_state:
        st.session_state.images = {}

    container_class = "full-container" + (" with-image" if "car_image" in st.session_state.images else "")

    st.markdown(
        """
        <style>
        .stApp {
            background-image: url(data:image/jpeg;base64,"""
        + background_image
        + """);
            background-attachment: fixed;
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            min-height: 100vh;
            position: relative;
        }

        .block-container {
            padding-top: 0;
            margin-top: 0;
        }

        .stApp::before {
            content: "";
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0, 0, 0, 0.7);
            z-index: 0;
        }

        .stApp {
            color: white;
        }

        .stButton>button {
            background-color: rgba(41, 128, 185, 0.9);
            color: white;
            border: none;
            padding: 0.5rem 2rem;
            margin-top: 30px;
            border-radius: 5px;
            cursor: pointer;
        }
        
        .stButton>button:disabled {
            background-color: rgba(41, 128, 185, 0.4);
            cursor: not-allowed;
        }

        .full-container {
            background-color: rgba(0, 0, 0, 0.75);
            width: 850px; 
            margin-bottom: 20px;
            padding: 100px 100px;
            min-height: 130vh;
            position: fixed;
            left: calc(50% - 425px);
            transform: none;  /* transform 제거 */
            backdrop-filter: blur(10px);
            box-shadow: 0 0 40px rgba(0, 0, 0, 0.5);
            border-radius: 20px;
            overflow: visible;
        }

        .block-container {
            padding-bottom: 100px;
        }

        .stImage {
            margin-bottom: 40px;
        }

        .full-container.with-image {
            min-height: 170vh;
        }

        .step-3-container {
            margin-top: 20px
            width: 100%;
        }

        .stTextInput > div > div > input,
        .stNumberInput > div > div > input,
        .stSelectbox select,
        .stSelectbox > div > div > div {
            background-color: rgba(31, 33, 37, 0.9) !important;
            color: white !important;
            border: none !important;
            border-radius: 5px;
        }

        .stSelectbox > div {
            background-color: rgba(31, 33, 37, 0.9) !important;
            color: white !important;
            border-radius: 5px;
        }

        .stSelectbox [data-baseweb="select"] > div {
            background-color: rgba(31, 33, 37, 0.9) !important;
            color: white !important;
        }

        .stSelectbox [data-baseweb="popover"] {
            background-color: rgba(31, 33, 37, 0.9) !important;
            color: white !important;
        }

        .stSelectbox [data-baseweb="option"],
        .stSelectbox [role="option"] {
            background-color: rgba(31, 33, 37, 0.9) !important;
            color: white !important;
        }

        .stSelectbox [data-baseweb="option"]:hover,
        .stSelectbox [role="option"]:hover {
            background-color: rgba(31, 33, 37, 0.9) !important;
            color: white !important;
        }

        .stSelectbox [aria-selected="true"] {
            background-color: rgba(41, 128, 185, 0.5) !important;
            color: white !important;
        }

        h1, h2, h3, p, label {
            color: white !important;
        }

        .input-label {
            font-size: 18px;
            font-weight: 500;
            color: white !important;
            margin-bottom: 8px;
        }

        .stRadio > div {
            display: flex !important;
            flex-direction: row !important;
            align-items: center !important;
            white-space: nowrap !important;
            gap: 0 !important;
        }

        .stRadio > div > div {
            display: flex !important;
            flex-direction: row !important;
            white-space: nowrap !important;
            margin-right: 10px !important;
        }

        .stRadio > div > div > label {
            white-space: nowrap !important;
            margin-right: 5px !important;
        }

        .preview-image-container {
            display: flex;
            justify-content: center;
            align-items: center;
            width: 100%;
            margin: 20px 0;
        }

        .preview-image-container > div {
            display: flex;
            flex-direction: column;
            align-items: center;
            width: 100%;
        }

        .preview-image-container img {
            margin: 0 auto;
        }

        .main-title {
            margin-top: 40px;  /* 상단 여백 추가 */
        }

        [data-testid="stSidebar"] {
            position: fixed !important;
            z-index: 1000;
        }

        /* number input과 selectbox 공통 스타일 */
        .stNumberInput > div > div > input, .stSelectbox > div > div {
            background-color: rgba(31, 33, 37, 0.9) !important;
            color: white !important;
        }
        
        /* selectbox 드롭다운 메뉴 스타일 */
        .stSelectbox > div > div > ul {
            background-color: rgba(31, 33, 37, 0.9) !important;
            color: white !important;
        }
        
        /* number input 버튼 스타일 */
        .stNumberInput > div > div > button {
            background-color: rgba(31, 33, 37, 0.9) !important;
            color: white !important;
        }

        /* 포커스 상태에서도 배경색 유지 */
        .stNumberInput > div > div > input:focus, .stSelectbox > div > div:focus {
            background-color: rgba(31, 33, 37, 0.9) !important;
            color: white !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="full-container">', unsafe_allow_html=True)
    st.markdown('<div style="height: 50px;"></div>', unsafe_allow_html=True)

    title_col, button_col = st.columns([3, 1])
    with title_col:
        st.title("차량 판매량 예측")
        st.subheader("Step 1: 필수 정보 입력")
    with button_col:
        all_inputs_ready = (
            len(st.session_state.images) == 1 
            and st.session_state.genmodel.strip() != "" 
            and st.session_state.entry_price > 0 
        )

        if all_inputs_ready:
            if st.button("Next →", key="next_button"):
                image_path = st.session_state.images.get("car_image")
                
                tabular_data = {}
                tabular_data["Genmodel"] = st.session_state.genmodel
                tabular_data["Entry_price"] = float(st.session_state.entry_price)

                categorical_fields = {
                    "Maker": st.session_state.maker,
                    "Color": st.session_state.color,
                    "Bodytype": st.session_state.bodytype,
                    "Fuel_type": st.session_state.fuel_type,
                    "Gearbox": st.session_state.gearbox,
                }
                for field, value in categorical_fields.items():
                    if value and value.strip():
                        tabular_data[field] = value

                numerical_fields = {
                    "Height": (
                        float(st.session_state.height)
                        if st.session_state.height
                        else None
                    ),
                    "Width": (
                        float(st.session_state.width)
                        if st.session_state.width
                        else None
                    ),
                    "Seat_num": (
                        int(st.session_state.seat_num)
                        if st.session_state.seat_num
                        else None
                    ),
                    "Door_num": (
                        int(st.session_state.door_num)
                        if st.session_state.door_num
                        else None
                    ),
                    "Year": (
                        int(st.session_state.year) if st.session_state.year else None
                    ),
                    "First_release_year": (
                        int(st.session_state.first_release_year)
                        if st.session_state.first_release_year
                        else None
                    ),
                    "Wheelbase": (
                        float(st.session_state.wheelbase)
                        if st.session_state.wheelbase
                        else None
                    ),
                    "Length": (
                        float(st.session_state.length)
                        if st.session_state.length
                        else None
                    ),
                    "Engine_size": (
                        float(st.session_state.engine_size)
                        if st.session_state.engine_size
                        else None
                    ),
                }
                for field, value in numerical_fields.items():
                    if value is not None and value != 0:
                        tabular_data[field] = value

                print("\n=== Inference로 전달되는 데이터 ===")
                print("실제 입력된 데이터만 포함된 tabular_data:")
                print(tabular_data)
                print("=" * 50)

                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                model_path = "./configs/models/model.pt"
                config_path = "./configs/config.yaml"

                config = load_config(config_path)
                model = load_model(model_path, device)

                image, categorical, continuous = preprocess_data(
                    image_path, tabular_data, config, device
                )

                predicted_sales_amt, denorm_continuous, decoded_categorical = (
                    infer_sales_amount_and_features(
                        model, image_path, tabular_data, config, device
                    )
                )

                if predicted_sales_amt is not None:
                    print("\n=== 예측 결과 ===")
                    print(f"예측 판매량: {predicted_sales_amt:,.0f} 대")
                    print("\n수치형 데이터:")
                    print(denorm_continuous)
                    print("\n범주형 데이터:")
                    print(decoded_categorical)

                    print("\n=== 예측에 사용된 값 ===")
                    print("입력된 tabular data:")
                    print(tabular_data)

                    st.session_state.predicted_sales_amt = predicted_sales_amt

                    predicted_values = {}

                    numerical_fields = [
                        "First_release_year",
                        "Year",
                        "Width",
                        "Height",
                        "Seat_num",
                        "Door_num",
                        "Wheelbase",
                        "Length",
                        "Engine_size",
                    ]

                    for field in numerical_fields:
                        if field not in tabular_data or not tabular_data[field]:
                            if field in denorm_continuous:
                                predicted_values[field] = denorm_continuous[field]

                    categorical_fields = [
                        "Maker",
                        "Color",
                        "Bodytype",
                        "Fuel_type",
                        "Gearbox",
                    ]
                    for field in categorical_fields:
                        if (
                            field not in tabular_data
                            or not tabular_data[field]
                            or tabular_data[field].strip() == ""
                        ):
                            if field in decoded_categorical:
                                predicted_values[field] = decoded_categorical[field]

                    if predicted_values:
                        st.session_state.predicted_values = predicted_values

                    st.switch_page("pages/result.py")
        else:
            st.button("Next →", disabled=True, key="next_button_disabled")

    genmodel_list = [""] + [
        v for v in config["categorical_mapping"]["Genmodel"].values()
    ]
    bodytype_list = get_values_from_mapping(config["categorical_mapping"]["Bodytype"])
    fuel_type_list = get_values_from_mapping(config["categorical_mapping"]["Fuel_type"])
    gearbox_list = get_values_from_mapping(config["categorical_mapping"]["Gearbox"])
    color_list = get_values_from_mapping(config["categorical_mapping"]["Color"])
    maker_list = get_values_from_mapping(config["categorical_mapping"]["Maker"])

    col1, col2 = st.columns(2)

    with col1:
        subcol1, subcol2 = st.columns([2, 1])

        with subcol1:
            st.markdown(
                '<p class="input-label">차량 모델명</p>', unsafe_allow_html=True
            )
            genmodel = st.selectbox(
                label="차량 모델명",
                options=genmodel_list,
                index=0,
                key="genmodel_input",
                label_visibility="collapsed",
            )
            st.session_state.genmodel = genmodel

        with subcol2:
            st.markdown('<p class="input-label">입력 방식</p>', unsafe_allow_html=True)
            input_method = st.radio(
                label="입력 방식",
                options=["선택하기", "직접 입력"],
                key="input_method",
                horizontal=True,
                label_visibility="collapsed",
            )

    with col2:
        st.markdown('<p class="input-label">진입 가격 ($)</p>', unsafe_allow_html=True)
        entry_price = st.number_input(
            label="진입 가격",
            value=(
                int(st.session_state.entry_price) if st.session_state.entry_price else 0
            ),
            step=1,
            format="%d",
            key="entry_price_input",
            label_visibility="collapsed",
        )
        st.session_state.entry_price = entry_price

    st.markdown("---")
    st.subheader("Step 2: 차량 이미지 업로드")
    uploaded_file = st.file_uploader(
        "차량 이미지 업로드",
        type=["jpg", "jpeg", "png"],
        key="car_image",
    )

    if uploaded_file:
        st.session_state.images["car_image"] = uploaded_file
        
        preview_image = Image.open(uploaded_file)
        
        col1, col2, col3 = st.columns([1, 2, 1]) 
        with col2:
            st.image(
                preview_image,
                caption="업로드된 이미지",
                width=300 
            )
        
        uploaded_file.seek(0)

    st.markdown("---")
    st.subheader("Step 3: 추가 정보 입력")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            '<p class="input-label">최초 출시 연도 (년)</p>', unsafe_allow_html=True
        )
        first_release_year = st.number_input(
            label="최초 출시 연도",
            value=(
                int(st.session_state.first_release_year)
                if st.session_state.first_release_year
                else 0
            ),
            step=1,
            format="%d",
            key="first_release_year_input",
            label_visibility="collapsed",
        )
        st.session_state.first_release_year = str(first_release_year)

        st.markdown(
            '<p class="input-label">출시 연도 (년)</p>', unsafe_allow_html=True
        )
        year = st.number_input(
            label="출시 연도",
            value=(int(st.session_state.year) if st.session_state.year else 0),
            step=1,
            format="%d",
            key="year_input",
            label_visibility="collapsed",
        )
        st.session_state.year = str(year)

        st.markdown('<p class="input-label">축거 (mm)</p>', unsafe_allow_html=True)
        wheelbase = st.number_input(
            label="축거",
            value=(
                float(st.session_state.wheelbase)
                if st.session_state.wheelbase
                else 0.0
            ),
            step=0.1,
            format="%.1f",
            key="wheelbase_input",
            label_visibility="collapsed",
        )
        st.session_state.wheelbase = str(wheelbase)

        st.markdown(
            '<p class="input-label">차량 너비 (mm)</p>', unsafe_allow_html=True
        )
        width = st.number_input(
            label="차량 너비",
            value=(
                float(st.session_state.width) if st.session_state.width else 0.0
            ),
            step=0.1,
            format="%.1f",
            key="width_input",
            label_visibility="collapsed",
        )
        st.session_state.width = str(width)

        st.markdown(
            '<p class="input-label">변속기 종류</p>', unsafe_allow_html=True
        )
        gearbox = st.selectbox(
            label="변속기 종류",
            options=gearbox_list,
            index=0,
            key="gearbox_input",
            label_visibility="collapsed",
        )
        st.session_state.gearbox = gearbox

    with col2:
        st.markdown(
            '<p class="input-label">차량 길이 (mm)</p>', unsafe_allow_html=True
        )
        length = st.number_input(
            label="차량 길이",
            value=(
                float(st.session_state.length) if st.session_state.length else 0.0
            ),
            step=0.1,
            format="%.1f",
            key="length_input",
            label_visibility="collapsed",
        )
        st.session_state.length = str(length)

        st.markdown(
            '<p class="input-label">차량 높이 (mm)</p>', unsafe_allow_html=True
        )
        height = st.number_input(
            label="차량 높이",
            value=(
                float(st.session_state.height) if st.session_state.height else 0.0
            ),
            step=0.1,
            format="%.1f",
            key="height_input",
            label_visibility="collapsed",
        )
        st.session_state.height = str(height)

        st.markdown(
            '<p class="input-label">엔진 배기량 (L)</p>', unsafe_allow_html=True
        )
        engine_size = st.number_input(
            label="엔진 배기량",
            value=(
                float(st.session_state.engine_size)
                if st.session_state.engine_size
                else 0.0
            ),
            step=0.1,
            format="%.1f",
            key="engine_size_input",
            label_visibility="collapsed",
        )
        st.session_state.engine_size = str(engine_size)

        st.markdown(
            '<p class="input-label">좌석 수 (개)</p>', unsafe_allow_html=True
        )
        seat_num = st.number_input(
            label="좌석 수",
            value=(
                int(st.session_state.seat_num) if st.session_state.seat_num else 0
            ),
            step=1,
            format="%d",
            key="seat_num_input",
            label_visibility="collapsed",
        )
        st.session_state.seat_num = str(seat_num)

        st.markdown('<p class="input-label">차량 색상</p>', unsafe_allow_html=True)
        color = st.selectbox(
            label="차량 색상",
            options=color_list,
            index=0,
            key="color_input",
            label_visibility="collapsed",
        )
        st.session_state.color = color

    with col3:
        st.markdown(
            '<p class="input-label">도어 수 (개)</p>', unsafe_allow_html=True
        )
        door_num = st.number_input(
            label="도어 수",
            value=(
                int(st.session_state.door_num) if st.session_state.door_num else 0
            ),
            step=1,
            format="%d",
            key="door_num_input",
            label_visibility="collapsed",
        )
        st.session_state.door_num = str(door_num)

        st.markdown('<p class="input-label">제조사</p>', unsafe_allow_html=True)
        maker = st.selectbox(
            label="제조사",
            options=maker_list,
            index=0,
            key="maker_input",
            label_visibility="collapsed",
        )
        st.session_state.maker = maker

        st.markdown('<p class="input-label">차체 유형</p>', unsafe_allow_html=True)
        bodytype = st.selectbox(
            label="차체 유형",
            options=bodytype_list,
            index=0,
            key="bodytype_input",
            label_visibility="collapsed",
        )
        st.session_state.bodytype = bodytype

        st.markdown('<p class="input-label">연료 유형</p>', unsafe_allow_html=True)
        fuel_type = st.selectbox(
            label="연료 유형",
            options=fuel_type_list,
            index=0,
            key="fuel_type_input",
            label_visibility="collapsed",
        )
        st.session_state.fuel_type = fuel_type

    st.markdown('</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    predict_page()

