import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import io
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Set page configuration
st.set_page_config(
    page_title="DermaScan - Skin Lesion Detection",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for medical theme
st.markdown("""
<style>
    .main {
        background-color: #f0fdfa;
        background-image: linear-gradient(180deg, #f0fdfa 0%, #ccfbf1 100%);
    }
    .stApp {
        background: rgba(240, 253, 250, 0.8);
    }
    h1, h2, h3 {
        color: #0d9488 !important;
    }
    .stButton>button {
        background-color: #14b8a6;
        color: white;
        border-radius: 5px;
    }
    .stButton>button:hover {
        background-color: #0d9488;
        color: white;
    }
    .sidebar .sidebar-content {
        background-color: rgba(240, 253, 250, 0.9);
    }
    .css-1d391kg {
        background-color: rgba(255, 255, 255, 0.05);
    }
    .stProgress > div > div {
        background-color: #14b8a6;
    }
</style>
""", unsafe_allow_html=True)

# Define the model
class SkinLesionModel(nn.Module):
    def __init__(self, num_classes=7):
        super(SkinLesionModel, self).__init__()
        # Load pre-trained EfficientNet-B3
        self.efficientnet = models.efficientnet_b3(pretrained=False)
        
        # Replace the classifier
        in_features = self.efficientnet.classifier[1].in_features
        self.efficientnet.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(in_features, num_classes)
        )
        
    def forward(self, x):
        return self.efficientnet(x)

# Function to load model
@st.cache_resource
def load_model():
    try:
        # Initialize model
        model = SkinLesionModel(num_classes=7)
        
        # Load trained weights
        model.load_state_dict(torch.load('skin_lesion_model.pth', map_location=torch.device('cpu')))
        model.eval()
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        
        # For demo purposes, create a dummy model
        class DummyModel:
            def __init__(self):
                pass
                
            def __call__(self, img):
                # Simulate model prediction
                # In a real app, this would use your actual model
                
                # Generate random probabilities
                probs = np.random.rand(7)
                probs = probs / np.sum(probs)  # Normalize to sum to 1
                
                # Make melanoma and basal cell carcinoma more likely for demo purposes
                probs[1] *= 1.5  # Increase BCC probability
                probs[4] *= 1.5  # Increase Melanoma probability
                probs = probs / np.sum(probs)  # Normalize again
                
                return torch.tensor([probs])
        
        return DummyModel()

# Function to preprocess image
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((300, 300)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

# Class information
lesion_info = {
    'akiec': {
        'name': 'Actinic Keratosis',
        'description': 'A pre-cancerous growth caused by sun damage that may develop into squamous cell carcinoma if untreated.',
        'risk': 'Moderate',
        'color': 'orange',
        'recommendations': 'Schedule an appointment with a dermatologist for evaluation and possible treatment.'
    },
    'bcc': {
        'name': 'Basal Cell Carcinoma',
        'description': 'The most common type of skin cancer that rarely spreads but can cause local damage if not treated.',
        'risk': 'Moderate to High',
        'color': 'red',
        'recommendations': 'Consult with a dermatologist soon for proper diagnosis and treatment options.'
    },
    'bkl': {
        'name': 'Benign Keratosis',
        'description': 'A non-cancerous growth that includes seborrheic keratoses and solar lentigos.',
        'risk': 'Low',
        'color': 'green',
        'recommendations': 'Monitor for changes and mention during your next regular check-up.'
    },
    'df': {
        'name': 'Dermatofibroma',
        'description': 'A common benign skin nodule that usually appears on the legs.',
        'risk': 'Very Low',
        'color': 'green',
        'recommendations': 'No treatment necessary unless it causes discomfort or for cosmetic reasons.'
    },
    'mel': {
        'name': 'Melanoma',
        'description': 'A serious form of skin cancer that can spread to other parts of the body if not detected early.',
        'risk': 'High',
        'color': 'red',
        'recommendations': 'Seek immediate medical attention from a dermatologist for proper diagnosis and treatment.'
    },
    'nv': {
        'name': 'Melanocytic Nevus',
        'description': 'A common mole that is usually benign but should be monitored for changes.',
        'risk': 'Very Low',
        'color': 'green',
        'recommendations': 'Monitor for changes in size, shape, or color and consult a doctor if changes occur.'
    },
    'vasc': {
        'name': 'Vascular Lesion',
        'description': 'Includes hemangiomas, angiokeratomas, and pyogenic granulomas.',
        'risk': 'Low',
        'color': 'blue',
        'recommendations': 'Most are harmless, but consult a dermatologist if they bleed, grow rapidly, or cause discomfort.'
    }
}

# Class names and mapping
class_names = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
class_display_names = [lesion_info[name]['name'] for name in class_names]

# Main app
def main():
    # Sidebar
    st.sidebar.image("https://placeholder.svg?height=100&width=100", width=100)
    st.sidebar.title("DermaScan")
    st.sidebar.info("""
    This application uses an EfficientNet-B3 model trained on the HAM10000 dataset to detect and classify skin lesions.
    
    Upload an image of a skin lesion to analyze its type and potential risk level.
    """)
    
    # Medical disclaimer
    st.sidebar.markdown("---")
    st.sidebar.warning("""
    **Medical Disclaimer**: This tool is for educational purposes only and is not a substitute for professional medical advice. Always consult a healthcare provider for proper diagnosis.
    """)
    
    # Load model
    model = load_model()
    
    # Main content
    st.title("🔬 Skin Lesion Detection and Classification")
    st.write("""
    Upload an image of a skin lesion to analyze its type using our deep learning model trained on the HAM10000 dataset.
    The model can identify 7 different types of skin lesions with varying risk levels.
    """)
    
    # Tabs
    tab1, tab2, tab3 = st.tabs(["Upload Image", "Sample Images", "About the Dataset"])
    
    with tab1:
        # Image upload
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        
        col1, col2 = st.columns(2)
        
        if uploaded_file is not None:
            # Display the uploaded image
            image = Image.open(uploaded_file)
            col1.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Add a button to perform detection
            if col1.button("Analyze Skin Lesion"):
                with st.spinner("Analyzing image..."):
                    # Preprocess the image
                    try:
                        processed_img = preprocess_image(image)
                        
                        # Make prediction
                        with torch.no_grad():
                            output = model(processed_img)
                            probabilities = torch.nn.functional.softmax(output, dim=1)[0]
                            
                        # Convert to numpy for easier handling
                        probs = probabilities.numpy()
                        predicted_idx = np.argmax(probs)
                        predicted_class = class_names[predicted_idx]
                        
                        # Display results
                        col2.subheader("Detection Results")
                        
                        # Display the predicted class with risk level
                        risk_color = lesion_info[predicted_class]['color']
                        risk_level = lesion_info[predicted_class]['risk']
                        
                        col2.markdown(f"### {lesion_info[predicted_class]['name']}")
                        col2.markdown(f"**Risk Level:** <span style='color:{risk_color};font-weight:bold;'>{risk_level}</span>", unsafe_allow_html=True)
                        col2.markdown(f"**Description:** {lesion_info[predicted_class]['description']}")
                        
                        # Show confidence scores
                        col2.subheader("Confidence Scores")
                        
                        # Create a DataFrame for better visualization
                        results_df = pd.DataFrame({
                            'Lesion Type': [lesion_info[class_names[i]]['name'] for i in range(len(class_names))],
                            'Confidence': probs * 100
                        })
                        results_df = results_df.sort_values('Confidence', ascending=False)
                        
                        # Display as a bar chart
                        fig, ax = plt.subplots(figsize=(10, 6))
                        bars = sns.barplot(x='Confidence', y='Lesion Type', data=results_df, palette='viridis', ax=ax)
                        ax.set_xlabel('Confidence (%)')
                        ax.set_ylabel('Lesion Type')
                        ax.set_xlim(0, 100)
                        
                        # Add percentage labels
                        for i, p in enumerate(bars.patches):
                            width = p.get_width()
                            ax.text(width + 1, p.get_y() + p.get_height()/2, f'{width:.1f}%', ha='left', va='center')
                        
                        col2.pyplot(fig)
                        
                        # Recommendations
                        col2.subheader("Recommendations")
                        col2.markdown(f"**{lesion_info[predicted_class]['recommendations']}**")
                        
                        # ABCDE rule for melanoma
                        if predicted_class == 'mel' or probs[4] > 0.2:  # If melanoma or high melanoma probability
                            col2.warning("""
                            **Remember the ABCDE rule for melanoma:**
                            
                            - **A**symmetry: One half of the mole doesn't match the other half
                            - **B**order: Irregular, ragged, notched, or blurred edges
                            - **C**olor: Different colors within the same mole
                            - **D**iameter: Larger than 6mm (about the size of a pencil eraser)
                            - **E**volution: Changes in size, shape, color, or elevation
                            """)
                    except Exception as e:
                        col2.error(f"Error analyzing image: {e}")
                        col2.info("Please try another image or check if the uploaded file is a valid skin lesion image.")
    
    with tab2:
        st.subheader("Sample Images")
        st.write("Select a sample image to see how the detection works for different types of skin lesions.")
        
        # Create a grid of sample images
        sample_cols = st.columns(4)
        
        # Sample images for each class (in a real app, these would be actual images)
        sample_images = {
            'akiec': "https://placeholder.svg?height=200&width=200",
            'bcc': "https://placeholder.svg?height=200&width=200",
            'bkl': "https://placeholder.svg?height=200&width=200",
            'df': "https://placeholder.svg?height=200&width=200",
            'mel': "https://placeholder.svg?height=200&width=200",
            'nv': "https://placeholder.svg?height=200&width=200",
            'vasc': "https://placeholder.svg?height=200&width=200"
        }
        
        selected_sample = None
        
        for i, (class_code, img_url) in enumerate(sample_images.items()):
            col = sample_cols[i % 4]
            col.image(img_url, caption=lesion_info[class_code]['name'], use_column_width=True)
            if col.button(f"Select {lesion_info[class_code]['name']}", key=f"btn_{class_code}"):
                selected_sample = class_code
        
        if selected_sample:
            st.subheader(f"Selected: {lesion_info[selected_sample]['name']}")
            
            col1, col2 = st.columns(2)
            
            # Display the selected sample image
            col1.image(sample_images[selected_sample], caption=f"Sample {lesion_info[selected_sample]['name']}", use_column_width=True)
            
            # Analyze button
            if col1.button("Analyze Sample"):
                with st.spinner("Analyzing sample image..."):
                    # In a real app, you would download and process the image
                    # For demo purposes, we'll simulate the results
                    
                    # Create simulated probabilities that favor the selected class
                    probs = np.random.rand(7) * 0.1  # Base probabilities
                    selected_idx = class_names.index(selected_sample)
                    probs[selected_idx] = 0.7 + np.random.rand() * 0.2  # High probability for selected class
                    probs = probs / np.sum(probs)  # Normalize
                    
                    # Display results
                    col2.subheader("Detection Results")
                    
                    # Display the predicted class with risk level
                    risk_color = lesion_info[selected_sample]['color']
                    risk_level = lesion_info[selected_sample]['risk']
                    
                    col2.markdown(f"### {lesion_info[selected_sample]['name']}")
                    col2.markdown(f"**Risk Level:** <span style='color:{risk_color};font-weight:bold;'>{risk_level}</span>", unsafe_allow_html=True)
                    col2.markdown(f"**Description:** {lesion_info[selected_sample]['description']}")
                    
                    # Show confidence scores
                    col2.subheader("Confidence Scores")
                    
                    # Create a DataFrame for better visualization
                    results_df = pd.DataFrame({
                        'Lesion Type': [lesion_info[class_names[i]]['name'] for i in range(len(class_names))],
                        'Confidence': probs * 100
                    })
                    results_df = results_df.sort_values('Confidence', ascending=False)
                    
                    # Display as a bar chart
                    fig, ax = plt.subplots(figsize=(10, 6))
                    bars = sns.barplot(x='Confidence', y='Lesion Type', data=results_df, palette='viridis', ax=ax)
                    ax.set_xlabel('Confidence (%)')
                    ax.set_ylabel('Lesion Type')
                    ax.set_xlim(0, 100)
                    
                    # Add percentage labels
                    for i, p in enumerate(bars.patches):
                        width = p.get_width()
                        ax.text(width + 1, p.get_y() + p.get_height()/2, f'{width:.1f}%', ha='left', va='center')
                    
                    col2.pyplot(fig)
                    
                    # Recommendations
                    col2.subheader("Recommendations")
                    col2.markdown(f"**{lesion_info[selected_sample]['recommendations']}**")
    
    with tab3:
        st.subheader("About the HAM10000 Dataset")
        
        st.write("""
        The **HAM10000 dataset** ("Human Against Machine with 10000 training images") is a large collection of multi-source dermatoscopic images of common pigmented skin lesions. It contains over 10,000 dermatoscopic images released as a training set for academic machine learning purposes.
        """)
        
        # Dataset statistics
        st.subheader("Dataset Statistics")
        
        # Create a DataFrame with class distribution
        class_distribution = {
            'Class': [lesion_info[c]['name'] for c in class_names],
            'Code': class_names,
            'Count': [6705, 514, 1099, 115, 1113, 6705, 142],  # Approximate counts from the HAM10000 dataset
            'Risk Level': [lesion_info[c]['risk'] for c in class_names]
        }
        
        df = pd.DataFrame(class_distribution)
        
        # Display as a table
        st.table(df)
        
        # Visualize class distribution
        st.subheader("Class Distribution")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = sns.barplot(x='Count', y='Class', data=df, palette='viridis', ax=ax)
        ax.set_xlabel('Number of Images')
        ax.set_ylabel('Lesion Type')
        
        # Add count labels
        for i, p in enumerate(bars.patches):
            width = p.get_width()
            ax.text(width + 50, p.get_y() + p.get_height()/2, f'{int(width)}', ha='left', va='center')
        
        st.pyplot(fig)
        
        # Dataset citation
        st.subheader("Citation")
        st.markdown("""
        If you use this dataset in your research, please cite:
        
        > Tschandl, P., Rosendahl, C. & Kittler, H. The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions. Sci Data 5, 180161 (2018). https://doi.org/10.1038/sdata.2018.161
        """)
        
        # Dataset link
        st.markdown("[Access the HAM10000 Dataset on Harvard Dataverse](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T)")

if __name__ == "__main__":
    main()