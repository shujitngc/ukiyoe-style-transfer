import streamlit as st
import requests
from PIL import Image
import io

st.title("Ukiyoe Style Transfer")
st.write("画像をアップロードすると、CycleGANで浮世絵風に変換します。")

uploaded_file = st.file_uploader(
    "画像を選択してください",
    type=["jpg", "jpeg", "png"]
)

model_type = st.selectbox(
    "変換方向を選択してください",
    ("g2z", "z2g")
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="元画像", use_column_width=True)

    if st.button("変換する"):
        uploaded_file.seek(0)
        bytes_data = uploaded_file.read()

        with st.spinner("変換中です..."):
            response = requests.post(
                "http://127.0.0.1:8000/predict",
                files={
                    "file": (
                        uploaded_file.name,
                        bytes_data,
                        uploaded_file.type
                    )
                },
                data={
                    "model": model_type
                }
            )

        if response.status_code == 200:
            transformed_image = Image.open(io.BytesIO(response.content))
            st.image(
                transformed_image,
                caption="変換後画像",
                use_column_width=True
            )

            st.download_button(
                label="変換後画像をダウンロード",
                data=response.content,
                file_name="ukiyoe_output.jpg",
                mime="image/jpeg"
            )

            st.success("変換が完了しました。")
        else:
            try:
                error_data = response.json()
                st.error(f"Error: {error_data.get('detail', error_data)}")
            except Exception:
                st.error(f"Error: {response.text}")