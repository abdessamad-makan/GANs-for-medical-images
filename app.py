import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import base64
from io import BytesIO

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Generator Model
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(100, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)

# CNN Classifier Model
class CNNClassifier(nn.Module):
    def __init__(self):
        super(CNNClassifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.fc1 = nn.Linear(32 * 8 * 8, 64)
        self.relu3 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = x.view(-1, 32 * 8 * 8)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

# Load GAN models
def load_gan_model(model_type):
    model = Generator().to(device)
    if model_type == 'Covid-19':
        model.load_state_dict(torch.load('netGvcovidfinalversion_dict.pth', map_location=device))
    else:
        model.load_state_dict(torch.load('netGnormalefinaldict.pth', map_location=device))
    model.eval()
    return model

# Load CNN model
def load_cnn_model():
    model = CNNClassifier().to(device)
    model.load_state_dict(torch.load('CNNfinal.pth', map_location=device))
    model.eval()
    return model

# Function to generate and display images
def generate_and_display_images(generator, num_images):
    with torch.no_grad():
        generator.eval()
        noise = torch.randn(num_images, 100, 1, 1, device=device)
        fake_images = generator(noise)
        for i in range(num_images):
            img = fake_images[i].cpu().numpy().transpose((1, 2, 0))
            img = ((img + 1) * 127.5).astype(np.uint8)
            pil_img = Image.fromarray(img)
            st.image(pil_img, caption=f'Generated Image {i+1}', use_column_width=False, width=250)
            # Provide download link for each generated image
            buffered = BytesIO()
            pil_img.save(buffered, format="JPEG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            st.markdown(
                f'<a href="data:file/txt;base64,{img_str}" download="generated_image_{i+1}.jpg" class="stButton"><button style="cursor:pointer;background-color:#007bff;border:none;border-radius:4px;color:white;padding:0.5rem 1rem;font-size:1rem;">Download Image {i+1}</button></a>',
                unsafe_allow_html=True
            )

# Function to classify an uploaded image
def classify_image(model, image):
    transform = transforms.Compose([
        transforms.Resize((32, 32)),  # Ensure the image is resized to 32x32
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    img = transform(image)
    img = img.unsqueeze(0)  # Add a batch dimension (BCHW format)
    with torch.no_grad():
        output = model(img)
    
    prediction = output.item()
    label = 'Normal' if prediction >= 0.5 else 'Covid-19'
    
    return label

def main():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Pacifico&display=swap');
        .custom-title {
            font-family: 'Lobster', cursive;
            color: black;
            font-size: 72px;
            text-align: center;
            margin-top: 50px;
            margin-bottom: 30px;
        }
        [data-testid="stAppViewContainer"] {
            background-size: cover;
        }
        .sidebar .sidebar-content {
            background: rgba(255, 255, 255, 0.9);
        }
        .stButton>button {
            cursor: pointer;
            background-color: #007bff;
            border: none;
            border-radius: 4px;
            color: white;
            padding: 0.5rem 1rem;
            font-size: 1rem;
            transition: all 0.3s ease-in-out;
        }
        .stButton>button:hover {
            background-color: #0056b3;
            transform: scale(1.05);
        }
        .css-1v3fvcr {
            font-family: 'Lobster', cursive;
        }
        [data-testid="stHeader"] {
            background-color: rgba(0, 0, 0, 0);
        }
        [data-testid="stSidebarContent"] {
        }
        .custom-text {
            font-size: 24px;
            font-family: 'Lobster', cursive;
            color: black;
        }
        </style>
         <h1 class="custom-title">Chest X-Ray Image Generator and Classifier</h1>
        """,
        unsafe_allow_html=True
    )

    st.sidebar.header('User Input')
    option = st.sidebar.selectbox('Choose an option', ('Generate Images', 'Classify Image'))

    if option == 'Generate Images':
        image_type = st.sidebar.selectbox('Choose Image Type', ('Normal', 'Covid-19'))
        num_images = st.sidebar.number_input('Number of Images to Generate', min_value=1, max_value=100, value=1)
        model = load_gan_model(image_type)

        if st.sidebar.button('Generate'):
            st.markdown(f'<div class="custom-text">Generated {num_images} {image_type} Images</div>', unsafe_allow_html=True)
            generate_and_display_images(model, num_images)

    elif option == 'Classify Image':
        uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        model = load_cnn_model()

        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, caption='Uploaded Image', use_column_width=False, width=300)
            st.write("")
            st.markdown('<div class="custom-text">Classifying...</div>', unsafe_allow_html=True)
            label = classify_image(model, image)
            st.markdown(f'<div class="custom-text">Prediction: {label}</div>', unsafe_allow_html=True)

if __name__ == '__main__':
    main()
