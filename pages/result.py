import streamlit as st
import torch
import base64

def get_image_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

background_image = get_image_base64(
    "./streamlit/parking_car_2.jpg"
)

def result_page():
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

        .full-container {
            background-color: rgba(0, 0, 0, 0.75);
            width: 850px; 
            margin-bottom: 20px;
            padding: 100px 100px;
            min-height: 130vh;
            position: fixed;
            left: calc(50% - 425px);  /* 컨테이너 너비의 절반만큼 왼쪽으로 이동 */
            transform: none;  /* transform 제거 */
            backdrop-filter: blur(10px);
            box-shadow: 0 0 40px rgba(0, 0, 0, 0.5);
            border-radius: 20px;
            overflow: visible;
        }

        .block-container {
            padding-top: 0;
            margin-top: 0;
        }

        .result-text {
            font-size: 24px;
            font-weight: 500;
            margin: 20px 0;
            color: white;
        }

        .prediction-value {
            color: white !important;
            font-size: 24px;
            font-weight: bold;
        }

        .predicted-value {
            color: #4CAF50 !important;
            font-size: 24px;
            font-weight: bold;
        }

        .prediction-label {
            color: #888;
            font-size: 14px;
            margin-bottom: 5px;
        }

        .info-container {
            background-color: rgba(41, 128, 185, 0.1);
            padding: 20px;
            border-radius: 10px;
            margin: 20px 0;
            border: 1px solid rgba(41, 128, 185, 0.3);
        }

        .stButton>button {
            background-color: rgba(41, 128, 185, 0.9);
            color: white;
            border: none;
            padding: 0.5rem 2rem;
            border-radius: 5px;
            cursor: pointer;
        }
        
        .stButton>button:hover {
            background-color: rgba(41, 128, 185, 1);
        }

        .image-container {
            margin: 30px 0;
            text-align: center;
        }
        .image-container img {
            max-width: 600px;
            border-radius: 10px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
        }

        .sales-container {
            display: flex;
            align-items: baseline;
            gap: 30px;
            margin: 20px 0;
            justify-content: flex-start;
        }

        .sales-label {
            color: white;
            font-size: 24px;
            font-weight: bold;
            min-width: 180px;
            line-height: 36px;
        }

        .sales-value {
            color: #4CAF50 !important;
            font-size: 30px;
            font-weight: bold;
            line-height: 36px;
        }

        .button-container {
            display: flex;
            justify-content: center;
            margin-top: 40px;
        }

        .stButton > button {
            background-color: rgba(41, 128, 185, 0.9);
            color: white;
            padding: 10px 30px;
            font-size: 16px;
            border-radius: 5px;
            border: none;
        }

        [data-testid="stSidebar"] {
            position: fixed !important;
            z-index: 1000;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="full-container">', unsafe_allow_html=True)

    st.markdown('<div style="height: 50px;"></div>', unsafe_allow_html=True)

    st.title("예측 결과")

    if "predicted_sales_amt" in st.session_state:
        sales_amount = int(st.session_state.predicted_sales_amt)
        st.markdown(
            f"""
            <div class="sales-container">
                <span class="sales-label">예상 연간 판매량:</span>
                <span class="sales-value">{sales_amount:,} 대</span>
            </div>
            """,
            unsafe_allow_html=True
        )

        if "images" in st.session_state and "car_image" in st.session_state.images:

            st.markdown('<div class="image-container">', unsafe_allow_html=True)
            st.image(
                st.session_state.images["car_image"],
                caption="업로드된 차량 이미지",
                use_container_width=True,
            )
            st.markdown("</div>", unsafe_allow_html=True)

        if "predicted_values" in st.session_state and st.session_state.predicted_values:
            cols = st.columns(3)

            all_fields = {
                'First_release_year': '최초 출시 연도 (년)',
                'Year': '출시 연도 (년)',
                'Height': '차량 높이 (mm)',
                'Seat_num': '좌석 수 (개)',
                'Wheelbase': '축거 (mm)',
                'Length': '차량 길이 (mm)',
                'Engine_size': '엔진 배기량 (L)',
                'Maker': '제조사',
                'Color': '차량 색상',
                'Bodytype': '차체 유형',
                'Fuel_type': '연료 유형',
                'Gearbox': '변속기 종류',
                'Door_num': '도어 수 (개)',
                'Width': '차량 너비 (mm)'
            }

            idx = 0
            for key, label in all_fields.items():
                col_idx = idx % 3
                with cols[col_idx]:
                    st.markdown(f'<div class="prediction-label">{label}</div>', unsafe_allow_html=True)
                    
                    input_value = getattr(st.session_state, key.lower().replace(' ', '_'), None)
                    if input_value and input_value != "0" and input_value != "0.0":
                        value = input_value
                        value_class = "prediction-value"

                    elif key in st.session_state.predicted_values:
                        value = st.session_state.predicted_values[key]
                        value_class = "predicted-value" 
                    else:
                        value = "N/A"
                        value_class = "prediction-value"

                    if isinstance(value, (int, float)):
                        if isinstance(value, float):
                            if key == 'Engine_size':
                                formatted_value = f"{value:,}"
                            else:
                                rounded_value = round(value)
                                formatted_value = f"{rounded_value:,}" 
                        else:
                            formatted_value = f"{value:,}"
                    else:
                        formatted_value = value

                    st.markdown(f'<div class="{value_class}">{formatted_value}</div>', unsafe_allow_html=True)
                idx += 1

    else:
        st.error("예측 결과를 찾을 수 없습니다.")

    st.markdown('<div style="height: 40px;"></div>', unsafe_allow_html=True)

    left_col, center_col, right_col = st.columns(3)
    with center_col:
        if st.button("새로운 예측하기"):
            st.switch_page("pages/predict.py")

    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    result_page()
