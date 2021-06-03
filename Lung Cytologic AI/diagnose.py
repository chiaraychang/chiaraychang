import torch
import torch.nn as nn
import numpy as np
from torchvision import datasets
import torchvision.transforms as trans
import torchvision.models as models
import time
import torch.backends.cudnn as cudnn
from albumentations.augmentations import transforms
from albumentations.core.composition import Compose
from HRFPN.HRFPN import *
import cv2
import torch.nn.functional as F
import os

test_on_gpu = torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(0)
cudnn.benchmark = True

transform_test = trans.Compose([
    trans.ToTensor(),
    trans.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])            
])

batch_size = 1
test_datapath = "./test_case_whole"
data_test  = datasets.ImageFolder(test_datapath,  transform=transform_test)
test_loader  = torch.utils.data.DataLoader(data_test,  batch_size=batch_size, shuffle=False, num_workers=4)

# classification model
model = models.resnet101(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)
savepath_ES   = "./models/resnet101_blur_cos_resume4.pth.tar"
checkpoint = torch.load(savepath_ES)
model.load_state_dict(checkpoint['model_state_dict'])
model = nn.Sequential(model, nn.Softmax(1))

# Segmentation model
model_hrfpn = HRFPN(False, classes=1)
model_hrfpn.load_state_dict(torch.load('./models/HRFPN_try3/model.pth'))

if test_on_gpu:
    print('testing on GPU')
    model = model.to(device)
    model_hrfpn = model_hrfpn.to(device)
model.eval()
model_hrfpn.eval()


def multi_scale_aug(image, rand_scale=1):
    base_size = 1024
    long_size = np.int(base_size * rand_scale + 0.5)
    h, w = image.shape[:2]
    if h > w:
        new_h = long_size
        new_w = np.int(w * long_size / h + 0.5)
    else:
        new_w = long_size
        new_h = np.int(h * long_size / w + 0.5)

    image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    return image


def multi_scale_inference(model, image, scales=[1]):
    batch, _, ori_height, ori_width = image.size()
    image = image.cpu().numpy()[0].transpose((1, 2, 0)).copy()
    
    final_pred = torch.zeros([1, 1, ori_height, ori_width]).cuda()

    for scale in scales:
        new_img = multi_scale_aug(image=image, rand_scale=scale)
        height, width = new_img.shape[:-1]
        
        new_img = new_img.transpose((2, 0, 1))
        new_img = np.expand_dims(new_img, axis=0)
        new_img = torch.from_numpy(new_img).to(device)

        preds = model(new_img)
        preds = preds[:, :, 0:height, 0:width]
        preds = F.interpolate(preds, (ori_height, ori_width), mode='bilinear', align_corners=False)
        final_pred += preds
        
    return final_pred / len(scales)


def segmentation(model, img, idx):
    top = int(round((img.size()[-2] - 1024) / 2.))
    left = int(round((img.size()[-1] - 1024) / 2.))
    img = img[..., top:top + 1024, left:left + 1024] / 255

    #output = model(img.to(device))
    output = multi_scale_inference(model, img, [0.875, 1, 1.25])
    output = torch.sigmoid(output).cpu().numpy()
    output[output<0.5] = 0
    output[output>=0.5]= 255
    cv2.imwrite(os.path.join('outputs', str(idx) + '.jpg'),
                (output[0, 0]).astype('uint8'))


class_correct = list(0. for i in range(2))
class_total = list(0. for i in range(2))
unfold = nn.Unfold((224,224), stride=112)

StartTime = time.time()
with torch.no_grad():
    for num, (data, target) in enumerate(test_loader):
        patches = unfold(data)
        patches = patches.permute(0,2,1).view(1, -1, 3, 224, 224).squeeze(0)
        
        if test_on_gpu:
            patches, target = patches.to(device), target.to(device)

        output = model(patches)
        prob, pred = torch.max(output, 1)
        
        pred_img = 0
        for num in range(len(prob)):
            if pred[num] == 1 and prob[num] >= 0.99:
                pred_img = 1
                segmentation(model_hrfpn, data, num)
                break

        correctness = int(pred_img == target[0].item()) 
        class_correct[target[0].item()] += correctness
        class_total[target[0].item()] += 1

EndTime = time.time()

classes = 2
for i in range(classes):
    if class_total[i] > 0:
        print('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (
            i, 100 * class_correct[i] / class_total[i],
            np.sum(class_correct[i]), np.sum(class_total[i])))
    else:
        print('Test Accuracy of %5s: N/A (no training examples)' % (i))

print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)' % (
    100. * np.sum(class_correct) / np.sum(class_total),
    np.sum(class_correct), np.sum(class_total)))

print('Time Usage: ', EndTime-StartTime, 'seconds')
print('FPS:', len(data_test)/(EndTime-StartTime))

