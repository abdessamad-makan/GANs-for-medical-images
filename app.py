import streamlit as st
import torch
import numpy as np
from PIL import Image
import torch.nn as nn
import streamlit as st
import torch
import os
import numpy as np
from torchvision.utils import save_image
from PIL import Image
import base64
from io import BytesIO
import streamlit as st
import torch
import os
import numpy as np
from torchvision.utils import save_image
from PIL import Image
import base64
from io import BytesIO
import zipfile

# Nomber de workers du dataloader
workers = 2
# Batch size
batch_size = 128
# size d'image
image_size = 64
# Nombre de channels j'ai utiliser les channels de couleur pour une bon image generer
nc = 3 
# Size de bruit z (la meme que de generator input)
nz = 100
# Size of feature maps in generator
ngf = 64
# Size of feature maps in discriminator
ndf = 64
# Learning rate de l'optimizateur
lr = 0.0002
# Beta1 hyperparameter de l'optimizateur Adam
beta1 = 0.5
# Number of GPUs available.
ngpu = 1
# numbre des epoch
num_epochs = 5
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")



# Generator Code
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.ngpu = 1
        self.main = nn.Sequential(
            nn.ConvTranspose2d( nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)



def load_gan_model():
    model = Generator()
    model.load_state_dict(torch.load('netG11.pth'))
    model.eval()  # Set the model to evaluation mode
    return model

# Function to generate and save images
def generate_and_save_images(generator, num_images):
    images = []

    with torch.no_grad():
        generator.eval()
        noise = torch.randn(num_images, 100, 1, 1, device='cpu')  # Adjust latent dimension if necessary
        fake_images = generator(noise)

    for i in range(num_images):
        img = fake_images[i].cpu().numpy().transpose((1, 2, 0))
        img = ((img + 1) * 127.5).astype(np.uint8)
        images.append(img)

        # Save the image temporarily to create a download button
        pil_img = Image.fromarray(img)
        buffered = BytesIO()
        pil_img.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode()

        # Display the image and a styled download button for each image
        st.image(img, caption=f'Generated Image {i+1}', use_column_width=True)
        st.markdown(
            f'<a href="data:file/txt;base64,{img_str}" download="generated_image_{i+1}.jpg" class="stButton"><button style="cursor:pointer;background-color:#007bff;border:none;border-radius:4px;color:white;padding:0.5rem 1rem;font-size:1rem;">Download Image {i+1}</button></a>',
            unsafe_allow_html=True
        )

    return images

def main():
    # Inject custom CSS for button styling
    st.markdown(
        """
        <style>
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
        </style>
        """,
        unsafe_allow_html=True
    )

    # Customize Streamlit page layout
    st.markdown(
        """
        <style>
        .reportview-container {
            background: linear-gradient(to bottom right, #f3f7f9, #dfe6ed);
        }
        .sidebar .sidebar-content {
            background: #f8f9fa;
        }
        .sidebar .sidebar-content .block-container {
            background: #fff;
        }
        .css-1v3fvcr {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.title('Chest X-Ray Image Generator')

    st.sidebar.header('User Input')
    num_images = st.sidebar.number_input('Number of Images to Generate', min_value=1, max_value=100, value=1)

    model = load_gan_model()

    if st.sidebar.button('Generate', key='generate_btn'):
        st.subheader(f'Generated {num_images} Images')
        generate_and_save_images(model, num_images)

if __name__ == '__main__':
    main()