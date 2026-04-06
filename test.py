import json
import os
from dashscope import MultiModalConversation
import base64
import mimetypes
import dashscope
import requests

# 以下为中国（北京）地域url，若使用新加坡地域的模型，需将url替换为：https://dashscope-intl.aliyuncs.com/api/v1
dashscope.base_http_api_url = 'https://dashscope.aliyuncs.com/api/v1'

# ---用于 Base64 编码 ---
# 格式为 data:{mime_type};base64,{base64_data}
def encode_file(file_path):
    mime_type, _ = mimetypes.guess_type(file_path)
    if not mime_type or not mime_type.startswith("image/"):
        raise ValueError("不支持或无法识别的图像格式")

    try:
        with open(file_path, "rb") as image_file:
            encoded_string = base64.b64encode(
                image_file.read()).decode('utf-8')
        return f"data:{mime_type};base64,{encoded_string}"
    except IOError as e:
        raise IOError(f"读取文件时出错: {file_path}, 错误: {str(e)}")

def download_image(image_url, save_path):
    r = requests.get(image_url, stream=True, timeout=300)
    r.raise_for_status()
    with open(save_path, "wb") as f:
        for chunk in r.iter_content(8192):
            if chunk:
                f.write(chunk)

# 获取图像的 Base64 编码
# 调用编码函数，请将 "/path/to/your/image.png" 替换为您的本地图片文件路径，否则无法运行
image = encode_file("resources/assets/logo.png")

messages = [
    {
        "role": "user",
        "content": [
            {"image": image},
            {"text": "在画面右下角石板路旁、靠近树干根部的位置，以浅灰墨色手写体题写一首七言绝句，字体为行楷风格，笔触自然流畅、略带飞白，大小适中（约占画面高度1/10），与整体水墨淡雅氛围协调。诗文内容为：“青石桥畔柳风轻， 素手拈花闭目听。 一水碧痕浮旧梦， 半篙烟雨入空舲。”诗句横向排列，四句分两行书写（前两句一行，后两句一行），末句“舲”字右下角钤一枚朱红小印，印文为“江南”二字篆书，尺寸约等于单字高度的1/3。"}
        ]
    }
]

# 新加坡和北京地域的API Key不同。获取API Key：https://help.aliyun.com/zh/model-studio/get-api-key
# 若没有配置环境变量，请用百炼 API Key 将下行替换为：api_key="sk-xxx"
api_key = "sk-82d9839ad7f046bf8214bacd2127448d"

# qwen-image-2.0系列、qwen-image-edit-max、qwen-image-edit-plus系列支持输出1-6张图片
response = MultiModalConversation.call(
    api_key=api_key,
    model="qwen-image-edit-max",
    messages=messages,
    stream=False,
    n=1,
    watermark=False,
    negative_prompt=" ",
    prompt_extend=False,
    size="1024*1024",
)

if response.status_code == 200:
    for i, content in enumerate(response.output.choices[0].message.content, 1):
        image_url = content["image"]
        download_image(image_url, f"output_{i}.png")
else:
    print(response.code, response.message)