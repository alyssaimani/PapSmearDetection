import torch
from torchvision import datasets, models, transforms
from flask import Flask, render_template, request, url_for
from flask_uploads import UploadSet, configure_uploads, IMAGES
import os
import numpy as np
import cv2
import ssl
from skimage.segmentation import mark_boundaries
# from LimeViz import get_image, get_pil_transform, batch_explaination
ssl._create_default_https_context = ssl._create_unverified_context

app = Flask(__name__)
photos = UploadSet('photos', IMAGES)

app.config['UPLOADED_PHOTOS_DEST'] = 'static'
configure_uploads(app, photos)

FIGUPLOAD = "phototopredict.jpg"
CAMFIG = "cam.jpg"
TRYTHIS = "./static/phototopredict.jpg"

@app.context_processor
def override_url_for():
    return dict(url_for=dated_url_for)


def dated_url_for(endpoint, **values):
    if endpoint == 'static':
        filename = values.get('filename', None)
        if filename:
            file_path = os.path.join(app.root_path,
                                     endpoint, filename)
            values['q'] = int(os.stat(file_path).st_mtime)
    return url_for(endpoint, **values)

@app.route('/',methods=['GET','POST'])
def home():
    return render_template('index.html')

@app.route('/upload',methods=['GET','POST'])
def upload():
    if request.method == 'POST' and 'photo' in request.files:
        if os.path.exists('static/phototopredict.jpg'):
            os.remove('static/phototopredict.jpg')

        filename = photos.save(request.files['photo'], name=FIGUPLOAD)
        # image = get_image('static/'+filename)
        # BASE_PATH = 'static/'
        # img_0 = get_image(BASE_PATH + 'COVID-1023.png')
        # img_1 = get_image(BASE_PATH + 'Lung_Opacity-1006.png')
        # img_2 = get_image(BASE_PATH + 'Viral Pneumonia-1003.png')
        # img_3 = get_image(BASE_PATH + 'Viral Pneumonia-1003.png')
        # img_all = [image]

        # pill_transform = get_pil_transform()
        # image_transf = [pill_transform(img) for img in img_all]
                
        # probs = batch_predict(image_transf)
        # print('prob',probs)
        # temps, masks = batch_explaination(image_transf)
        # result=[]
        # for i in range(len(temps)):
        #     marked_img = mark_boundaries(temps[i], masks[i])
        #     conv_img = (marked_img * 255).astype('uint8')
        #     bgr_img =  cv2.cvtColor(conv_img, cv2.COLOR_RGB2BGR)
        #     result.append(bgr_img)
        # cv2.imwrite(os.path.join('static', "cam.jpg"), result[0])

        # Pap Smear
        # load images
        cropped_images = get_cropped_images('static/'+ filename)
        

        # model.load_state_dict(torch.load("deployed_asset/22-08-01_12-51-21_normal_best_model.pth",map_location=device))
        # model.train(False)
        # model.to(device)


        # image = Image.open("static/phototopredict.jpg")
        # image = image.convert('RGB')
        # image = data_transforms['val'](image)
        # image = image.unsqueeze(0).to(device)

        # # outputs = model(image)
        # # img_t = get_input_tensors(image)
        # outputs = model(image)
        # print(outputs)
        # _, preds = torch.max(outputs, 1)
        # probs = outputs.cpu().data.numpy()
        # # print(probs)

        # _, idx = outputs.sort(1,True)

        # # print([i for i in idx if probs[i.cpu().data.numpy()]>=THRESHOLD])
        # # return 0

        # for i in range(cfg_data["num_class"]):
        #     probs[0][i] = round(probs[0][i]*100, 2)

        # visualize_gradcam(model,device,image,"static/phototopredict.jpg",idx)
        # best_class = preds.cpu().data.numpy()[0]
    return render_template('upload.html', results = probs[0],filename = filename,threshold = float(request.form.get('threshold')))

    # dump(mean)
    # return "Upload - " + str(mean)

# Utilities
def initialize_model(cfg_data):
    if cfg_data["MODEL_ARCH"] == "resnet50":
        from models.ResNet50 import ResNet50_Model as net
        return net(cfg_data)
    if cfg_data["MODEL_ARCH"] == "dino_vits16":
        from models.DinoVits16 import DinoVits16_Model as dino
        return dino(cfg_data)
    return None
        

def visualize_gradcam(model,device,image,raw_image_path,idx):
    f_model = torch.nn.Sequential(*(list(model.model.children())[0:-2]))
    f_model.to(device)

    params = list(model.model.fc.parameters())
    weight = np.squeeze(params[0].data.cpu().numpy())
    bias = np.squeeze(params[1].data.cpu().numpy())

    feature_maps = f_model(image)
    CAMs = return_CAM(feature_maps.detach().cpu().numpy(), weight, [idx.cpu().numpy()[0]]) # generate the CAM for the input image
    
    heatmap = cv2.applyColorMap(CAMs[0], cv2.COLORMAP_JET)

    image = get_image(raw_image_path)
    # image = Image.open(raw_image_path)
    # image = image.convert('RGB')
    # image = np.array(image)
    # temps, masks = batch_explanation([image], model)

    # result = mark_boundaries(temps[0], masks[0])
    # result = 0.5 * heatmap + 0.5 * image

    # if os.path.exists(os.path.join('static',CAMFIG)):
    #     os.remove(os.path.join('static',CAMFIG))

    # cv2.imwrite(os.path.join('static',CAMFIG), result)


def return_CAM(feature_conv, weight, class_idx):
    """
    return_CAM generates the CAMs and up-sample it to 224x224
    arguments:
    feature_conv: the feature maps of the last convolutional layer
    weight: the weights that have been extracted from the trained parameters
    class_idx: the label of the class which has the highest probability
    """
    size_upsample = (224, 224)
    
    # we only consider one input image at a time, therefore in the case of 
    # VGG16, the shape is (1, 512, 7, 7)
    bz, nc, h, w = feature_conv.shape 
    output_cam = []
    for idx in class_idx[0]:
        beforeDot =  feature_conv.reshape((nc, h*w))# -> (512, 49)
        print(beforeDot.shape)
        cam = np.matmul(weight[idx], beforeDot) # -> (1, 512) x (512, 49) = (1, 49)
        print(cam.shape)
        cam = cam.reshape(h, w) # -> (7 ,7)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        cam_img = np.uint8(255 * cam_img)
        output_cam.append(cv2.resize(cam_img, size_upsample))
    return output_cam

if __name__ == '__main__':
    app.run(debug=True, port=8051)