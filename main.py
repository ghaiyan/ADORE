from flask import Flask, render_template, request, send_file
from PIL import Image
import torch
import torchvision.transforms as transforms
import numpy as np
import os
import base64


app = Flask(__name__)

# 图像预处理和转换
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # 获取上传的图像文件和模型选择
        image_file = request.files['image']
        model_name = request.form['model']

        # 模型文件路径
        model_path = os.path.join('model', model_name)

        # 加载模型
        model = torch.load(model_path, map_location=torch.device('cpu'))
        model.eval()

        # 获取上传的图像
        
        image_path = os.path.join('static', image_file.filename)
        image_file.save(image_path)
        imagename=image_file.filename

        # 进行图像分类
        # 预处理图像
        image = Image.open(image_file).convert('RGB')
        input_tensor = preprocess(image)
        input_batch = input_tensor.unsqueeze(0)

        # 将图像传递给模型进行预测
        with torch.no_grad():
            output = model(input_batch)

        # 计算置信度
        probabilities = torch.softmax(output, dim=1)[0]
        confidence = np.max(probabilities.numpy())

        # 获取预测标签
        _, predicted_idx = torch.max(output, 1)
        predicted_label = predicted_idx.item()

   
        

        return render_template('result.html', model=model_name, predicted_label=predicted_label, confidence=confidence,imagename=imagename)
    else:
        # 获取所有可用的模型列表
        model_files = os.listdir('model')
        model_names = [name for name in model_files if name.endswith('.pth')]

        return render_template('index.html', model_names=model_names)

@app.route('/download_sample')
def download_sample():
    # 提供样本文件的下载
    sample_file = 'sample/corrosionLevel1_660.jpg'
    return send_file(sample_file, as_attachment=True)

@app.route('/download_sample1')
def download_sample1():
    # 提供样本文件的下载
    sample_file = 'sample/corrosionLevel2_660.jpg'
    return send_file(sample_file, as_attachment=True)

@app.route('/download_sample2')
def download_sample2():
    # 提供样本文件的下载
    sample_file = 'sample/corrosionLevel3_660.jpg'
    return send_file(sample_file, as_attachment=True)

@app.route('/download_sample3')
def download_sample3():
    # 提供样本文件的下载
    sample_file = 'sample/corrosionLevel4_660.jpg'
    return send_file(sample_file, as_attachment=True)

@app.route('/download_sample4')
def download_sample4():
    # 提供样本文件的下载
    sample_file = 'sample/corrosionLevel5_660.jpg'
    return send_file(sample_file, as_attachment=True)
if __name__ == '__main__':
    app.run()
