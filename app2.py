import streamlit as st
import torch
from networks import define_G  # 确保你有定义 UnetGenerator 类或相关模型
from PIL import Image
from torchvision import transforms


# 函数: 加载 Pix2Pix 模型
def load_pix2pix_model(model_path, input_nc=3, output_nc=3, ngf=64, netG='unet_256'):
    # 实例化生成器模型
    generator = define_G(input_nc, output_nc, ngf, netG)
    # 加载预训练模型
    try:
        generator.load_state_dict(torch.load(model_path))
        generator.eval()
        print(f"Successfully loaded the Pix2Pix model from {model_path}")
    except RuntimeError as e:
        print(f"Error loading state_dict: {e}")
    return generator


# 定义图像变换
def get_transform():
    return transforms.Compose([
        transforms.Resize((256, 256)),  # 需要根据模型训练时的输入尺寸进行调整
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 归一化处理
    ])


# 定义反向变换
def get_inv_transform():
    return transforms.Compose([
        transforms.Normalize(mean=[-1, -1, -1], std=[2, 2, 2]),  # 反归一化处理
        transforms.ToPILImage()
    ])


# 函数: 使用 Pix2Pix 模型生成图像
def generate_image(generator, image_path):
    transform = get_transform()
    inv_transform = get_inv_transform()

    # 打开并转换图像
    input_image = Image.open(image_path).convert('RGB')
    input_tensor = transform(input_image).unsqueeze(0)  # 增加批次维度

    # 生成新图像
    with torch.no_grad():
        output_tensor = generator(input_tensor)[0]

    # 将输出张量转换为图像
    output_image = inv_transform(output_tensor)

    # 确保图像正常
    if output_image.mode != 'RGB':
        output_image = output_image.convert('RGB')

    return output_image


# 加载 Pix2Pix 模型
model_path = './checkpoints/facades_label2photo_pretrained/latest_net_G.pth'  # 替换为你选择的最佳生成器模型路径
generator = load_pix2pix_model(model_path)

# 创建Streamlit界面
st.title("Pix2Pix Image Generator")
st.write("Upload an image and the Pix2Pix model will generate a new image.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    input_image = Image.open(uploaded_file).convert('RGB')
    st.image(input_image, caption='Uploaded Image', use_column_width=True)

    # 生成新图像
    output_image = generate_image(generator, uploaded_file)

    st.image(output_image, caption='Generated Image', use_column_width=True)
