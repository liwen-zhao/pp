#部署反转效果，已实现

import os
import numpy as np
import streamlit as st
from PIL import Image
import time  # 确保导入了 time 模块以使用 sleep 函数

def main():
    # 确保必要的文件夹存在
    if not os.path.exists('static'):
        os.makedirs('static')
    if not os.path.exists('static/edit'):
        os.makedirs('static/edit')

    prompt_container = st.empty()  # 定义一个空容器
    if 'load_models' not in st.session_state:
        prompt_container.caption('Loading...')
        st.session_state.load_models = []

        # 在这里load模型，然后把模型传入寄存器
        st.session_state.load_models.append('./checkpoints/facades_label2photo_pretrained/latest_net_G.pth')
        # st.session_state.load_models.append(模型2)

        prompt_container.empty()  # 重新置为空容器，caption消失

    # 我用这个状态寄存器来防止模型调用过程中反复调用
    if 'editing' not in st.session_state:
        st.session_state.editing = False

    #  main app body
    st.markdown(
        """
        Upload an image and the model will generate a new image...
        """
    )

    # 上传图片
    image_container = st.empty()
    in_image = image_container.file_uploader("1.Input image:", type=["png", "jpg"])
    get_value = lambda x: x if x is None or isinstance(x, str) else x.getvalue()

    if 'input_img' not in st.session_state or get_value(st.session_state.input_img) != get_value(in_image):
        if 'input_img' in st.session_state and get_value(st.session_state.input_img) != get_value(in_image):
            print("update img...")
            for key in st.session_state.keys():
                # 只有加载好的模型存下来不被删除，其他寄存器内容被删
                if key != 'load_models':
                    del st.session_state[key]
            time.sleep(1)  # 确保在删除寄存器后等待1秒以完成文件处理

        st.session_state.input_img = in_image

        if in_image is not None:
            # 保存用户上传的图片
            image = Image.open(in_image).convert('RGB')
            image.save("static/111111.jpg")

    st.markdown('2.click the button to start:')
    img_edit_button = st.button('Start')

    # 这里写按下按键后的操作
    if img_edit_button:
        if not st.session_state.editing:
            st.session_state.editing = True
            print("user choosing done!editing...")

            # 此处调用模型
            # 假设我们调用模型进行处理并将结果保存到'edit/222.jpg'
            # 你需要将以下部分替换为实际的模型处理逻辑
            # Example:
            # model_output = process_image("static/111111.jpg")
            # model_output.save("static/edit/222.jpg")

            # 模拟处理结果，实际使用中要调用你自己的模型处理函数
            from PIL import ImageOps
            img = Image.open("static/111111.jpg")
            # 将原图反转为示例输出图像
            processed_img = ImageOps.invert(img.convert("RGB"))
            processed_img.save("static/edit/222.jpg")

            # 两列展示原图和编辑后的图
            col1, col2 = st.columns(2)
            with col1:
                st.subheader('your input:')
                ori_img = np.array(Image.open('static/111111.jpg'))
                st.image(ori_img)
            with col2:
                st.subheader('edited result:')
                return_img = np.array(Image.open('static/edit/222.jpg'))
                st.image(return_img)

            st.session_state.editing = False

if __name__ == "__main__":
    st.set_page_config(
        page_title="Color Reversal System", page_icon=":pencil2:"
    )
    st.title("--Color Reversal System--")
    main()
