import streamlit as st
import cv2
from ultralytics import YOLO
import tempfile
import os
from collections import defaultdict
import numpy as np

def main():
    st.set_page_config(page_title="Segmentation AI model", layout="wide")
    
    st.title("Project Instance Segmentation")
    st.write('<p style="font-size: 24px; font-weight: bold;">ทดสอบปัญญาประดิษฐ์ตรวจจับและแยกส่วนภาพหอยศัตรูพืช</p>', unsafe_allow_html=True)
    st.sidebar.header("งานวิจัยของศุภกร วงษ์เรืองพิบูล")

    model_path = "Image_GASInstSegm.pt"
    st.sidebar.success("Model loaded successfully...")
    uploaded_file = st.sidebar.file_uploader("เลือกไฟล์ภาพ...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:

        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
        tfile.write(uploaded_file.read())
        model = YOLO(model_path)
        img = cv2.imread(tfile.name)
        
        st.sidebar.info("กำลังประมวลผลภาพ (Segmenting)...")
        results = model(img)

        res_plotted = results[0].plot()
        res_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
        label_count = defaultdict(int)

        if results[0].boxes:
            for box in results[0].boxes:
                cls = int(box.cls[0])
                label = model.names[cls]
                label_count[label] += 1
            st.image(res_rgb, caption="ผลการทำ Instance Segmentation", width=640)

        st.subheader("สรุปผลการตรวจจับและแบ่งส่วน")
        if label_count:
            for label, count in label_count.items():
                st.write(f"- **{label}**: {count} ตัว/ชิ้น")
        else:
            st.write("ไม่พบวัตถุที่สนใจในภาพ")

        st.success("Segmentation Completed!")

        try:
            tfile.close()
            os.unlink(tfile.name)
        except Exception as e:
            st.error(f"เกิดข้อผิดพลาดในการลบไฟล์ชั่วคราว: {e}")

if __name__ == "__main__":
    main()