import os
import numpy as np
import cv2
import joblib
import copy
import torch
import torch.nn.functional as F
import matplotlib.image as mi

from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from glob import glob
from scipy import io
from torchvision import models, transforms
from pathology_ai_model.utils import get_modelpath


def standard_scale(data, testingflag, scaler_path):
    if testingflag:
        scaler = joblib.load(scaler_path)
    else:
        scaler = StandardScaler()
        scaler.fit(data)

        print(scaler.mean_)
        print(scaler.var_)
        joblib.dump(scaler, os.path.join(scaler_path))

    return scaler.transform(data)


def load_img(imagename, transform=None):
    # img = Image.open(imagename)
    img = cv2.imread(imagename, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.float32(img/255.)
    if transform is not None:
        img = transform(img)
    splitpart = imagename.split('_')
    return img, splitpart[-4], splitpart[-3] #return img, and position of H and W


def slidePredVis(root, preprocess, net, savepath, fac):
    imglist = glob(os.path.join(root, '*png'))
    maxH = max([int(i.split('_')[-2]) for i in imglist])//fac + 1
    maxW = max([int(i.split('_')[-1].replace(".png",'')) for i in imglist])//fac + 1
    slidescore = np.zeros([maxH, maxW])
    print(slidescore.shape)

    for imgname in tqdm(imglist):
        img, h, w = load_img(imgname, preprocess)
        with torch.no_grad():
            img = img.unsqueeze(0).cuda()
            # print(img.size())
            prob = F.softmax(net(img), 1, _stacklevel=5).cpu().numpy()[:, 1]
            # score = np.concatenate((score, prob.cpu().numpy()[:, 1]), axis=0)
            slidescore[(int(h)-1)//fac, (int(w)-1)//fac] = prob
    
    savename = os.path.join(savepath, root.split('/')[-1] + '_Vis')
    np.save(savename    + '.npy', slidescore)
    io.savemat(savename + '.mat', {'pred':slidescore})
    mi.imsave(savename  + '.png', slidescore, cmap='jet')


def slide_pred_vis_tumor(root, preprocess, net, npz_path, fac, scaler_path):
    net = net.eval()
    tumor_detect_npz = np.load(npz_path)
    scores, predictions, namelist = tumor_detect_npz['score'], tumor_detect_npz['bin'], tumor_detect_npz['namelist']
    tumornamelist = [namelist[i].split('/')[-1] + '.png' for i in range(len(namelist)) if predictions[i] == 0.0]

    imglist = glob(os.path.join(root, '*png'))
    maxH = max([int(i.split('_')[-4]) for i in imglist]) // fac + 1
    maxW = max([int(i.split('_')[-3].replace(".png", '')) for i in imglist]) // fac + 1
    slidescore = np.full([maxH, maxW],-1,dtype=float)

    print(slidescore.shape)

    all_h = np.array([])
    all_w = np.array([])
    all_score = np.array([])

    for imgname in tqdm(imglist):
        img, h, w = load_img(imgname, preprocess)

        with torch.no_grad():
            img = img.unsqueeze(0).cuda()
            # print(img.size())
            if imgname.split('/')[-1] in tumornamelist:
                prob = F.softmax(net(img), 1, _stacklevel=5).cpu().numpy()[:, 1]

                all_h = np.append(all_h, h)
                all_w = np.append(all_w, w)
                all_score = np.append(all_score, prob.flatten())

    all_score_scaled = standard_scale(data=all_score.reshape(-1, 1), testingflag=True, scaler_path=scaler_path).flatten()
    score_min, score_max = min(all_score_scaled), max(all_score_scaled)
    all_score_scaled_minmaxed = (all_score_scaled - score_min) / (score_max - score_min)

    for i in range(len(all_score_scaled_minmaxed)):
        slidescore[(int(all_h[i]) - 1) // fac, (int(all_w[i]) - 1) // fac] = all_score_scaled_minmaxed[i]

    return slidescore

def save_heatmap_file(root, savepath, slidescore):
    savename = os.path.join(savepath, root.split('\\')[-1] + 'tumor_Vis')
    np.save(savename + '.npy', slidescore)
    io.savemat(savename + '.mat', {'pred': slidescore})
    # mi.imsave(savename + '.png', slidescore, cmap='jet')

def postprocess(slidescore):
    temp = copy.deepcopy(slidescore)
    temp[slidescore == -1] = 1  ## 1 for non_tumor area
    temp[slidescore != -1] = 0  ## 0 for tumor area

    oppotemp = copy.deepcopy(slidescore)
    oppotemp[slidescore == -1] = 0  ## 0 for non_tumor area
    oppotemp[slidescore != -1] = 1  ## 1 for tumor area

    slidescore[slidescore == -1] = 0
    probmap = cv2.applyColorMap(np.uint8(255 * slidescore[:, :, None]), cv2.COLORMAP_JET)
    overlap = probmap * oppotemp[:, :, None] + temp[:, :, None] * 255

    return overlap

def make_heatmap(datapath, sampling_file, featsfile, root_dir, model_type, num_classes=2, gpu="0", norm_mean=[0.8201, 0.5207, 0.7189], norm_std=[0.1526, 0.1542, 0.1183]):
    preprocess = transforms.Compose([
        transforms.ToTensor(),  # operated on original image, rewrite on previous transform.
        transforms.Normalize(norm_mean, norm_std)])

    modelpath = get_modelpath(model_type + '.pkl')

    if len(gpu) > 1:
        net = torch.nn.DataParallel(models.resnet18(pretrained=False, num_classes=num_classes)).cuda()
        net.load_state_dict(torch.load(modelpath)) # load the finetune weight parameters
    else:
        net = models.resnet18(pretrained=False, num_classes=num_classes).cuda()
        net.load_state_dict({k.replace('module.',''):v for k,v in torch.load(modelpath).items()})

    with open(sampling_file, 'r') as fh:
      id_list = [line.rstrip().split('\\')[0] for line in fh]

    scaler_path = ''

    uid = np.unique(np.array(id_list))
    print('{} slides wait to be predicted!'.format(len(uid)))

    for id in uid:
        factor = 4

        savepath = os.path.join(root_dir, model_type, 'slide_heatmap', id)
        if not os.path.exists(savepath):
            os.makedirs(savepath)

        slidescore = slide_pred_vis_tumor(os.path.join(datapath, id), preprocess, net, featsfile, factor, scaler_path)

        print('highest:',np.max(slidescore) ,'lowest',np.min(slidescore[slidescore!=-1]))
        print('slidescore calculation finished')
        slidescore = cv2.resize(slidescore, dsize=None, fx=16, fy=16, interpolation=cv2.INTER_NEAREST)
        print('slide resize finished')
        outputfile = os.path.join(savepath, id)
        save_heatmap_file(os.path.join(datapath, id), outputfile, slidescore)

        overlap = postprocess(slidescore)
        cv2.imwrite(savepath + '/overlap.png', overlap)
