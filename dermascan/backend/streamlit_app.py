import streamlit as st
import requests
from PIL import Image
import io
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Set page configuration
st.set_page_config(
    page_title="DermaScan - Skin Lesion Detection",
    page_icon="🔬",
    layout="wide"
)

# Class information
lesion_info = {
    'akiec': {
        'name': 'Actinic Keratosis',
        'description': 'A pre-cancerous growth caused by sun damage that may develop into squamous cell carcinoma if untreated.',
        'risk': 'Moderate',
        'color': 'orange'
    },
    'bcc': {
        'name': 'Basal Cell Carcinoma',
        'description': 'The most common type of skin cancer that rarely spreads but can cause local damage if not treated.',
        'risk': 'Moderate to High',
        'color': 'red'
    },
    'bkl': {
        'name': 'Benign Keratosis',
        'description': 'A non-cancerous growth that includes seborrheic keratoses and solar lentigos.',
        'risk': 'Low',
        'color': 'green'
    },
    'df': {
        'name': 'Dermatofibroma',
        'description': 'A common benign skin nodule that usually appears on the legs.',
        'risk': 'Very Low',
        'color': 'green'
    },
    'mel': {
        'name': 'Melanoma',
        'description': 'A serious form of skin cancer that can spread to other parts of the body if not detected early.',
        'risk': 'High',
        'color': 'red'
    },
    'nv': {
        'name': 'Melanocytic Nevus',
        'description': 'A common mole that is usually benign but should be monitored for changes.',
        'risk': 'Very Low',
        'color': 'green'
    },
    'vasc': {
        'name': 'Vascular Lesion',
        'description': 'Includes hemangiomas, angiokeratomas, and pyogenic granulomas.',
        'risk': 'Low',
        'color': 'blue'
    }
}

def main():
    st.title("DermaScan: Skin Lesion Classifier")
    st.write("Upload an image of a skin lesion for analysis.")

    # Medical disclaimer
    st.warning("**Medical Disclaimer**: This tool is for educational purposes only and is not a substitute for professional medical advice.")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        if st.button("Analyze Skin Lesion"):
            try:
                with st.spinner("Analyzing..."):
                    # Prepare the file for the API request
                    files = {"file": uploaded_file.getvalue()}
                    
                    # Make the API request
                    response = requests.post("http://localhost:8502/api/predict", files=files)
                    
                    if response.status_code == 200:
                        result = response.json()
                        prediction = result['prediction']
                        confidences = result['confidences']
                        details = result.get('details', [])
                        
                        # Display the results
                        st.success(f"Prediction: {lesion_info[prediction]['name']}")
                        
                        # Display risk level
                        risk_color = lesion_info[prediction]['color']
                        risk_level = lesion_info[prediction]['risk']
                        st.markdown(f"**Risk Level:** <span style='color:{risk_color};font-weight:bold;'>{risk_level}</span>", unsafe_allow_html=True)
                        
                        # Display description
                        st.markdown(f"**Description:** {lesion_info[prediction]['description']}")
                        
                        # Display confidence scores
                        st.subheader("Confidence Scores")
                        
                        # Sort confidences by value
                        sorted_confidences = sorted(confidences.items(), key=lambda x: x[1], reverse=True)
                        
                        # Create a bar chart
                        labels = [lesion_info[class_name]['name'] for class_name, _ in sorted_confidences]
                        values = [conf * 100 for _, conf in sorted_confidences]
                        
                        fig, ax = plt.subplots(figsize=(10, 6))
                        bars = ax.barh(labels, values, color='teal')
                        ax.set_xlabel('Confidence (%)')
                        ax.set_xlim(0, 100)
                        
                        # Add percentage labels
                        for i, bar in enumerate(bars):
                            width = bar.get_width()
                            ax.text(width + 1, bar.get_y() + bar.get_height()/2, 
                                   f'{width:.1f}%', ha='left', va='center')
                        
                        st.pyplot(fig)
                        
                        # Display analysis details
                        if details:
                            st.subheader("Analysis Details")
                            for detail in details:
                                st.write(detail)
                    else:
                        st.error(f"Error: API returned status code {response.status_code}")
                        st.write(response.text)
            except Exception as e:
                st.error(f"Error: {e}")
                st.info("Make sure the API server is running on port 8502. Run 'python api.py' in a separate terminal.")

if __name__ == "__main__":
    main()
