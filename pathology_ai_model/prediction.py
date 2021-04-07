import os
import numpy as np
import random
import torch
from torch.utils.data import DataLoader
from torchvision import models, transforms

from pathology_ai_model.data_reader import ClsDataset
from pathology_ai_model.utils import get_modelpath, net_prediction_oneshop, patient_res_m3_oneshop, save_temp_excel_oneshop

def del_file(path):
    ls = os.listdir(path)
    for i in ls:
        c_path = os.path.join(path, i)
        if os.path.isdir(c_path):
            del_file(c_path)
        else:
            os.remove(c_path)

def start_model(datapath, sampling_file, result_path, seed=2020, gpu="0", net="resnet18", num_classes=2, num_workers=4, batch_size=256, norm_mean=[0.8201, 0.5207, 0.7189], norm_std=[0.1526, 0.1542, 0.1183]):
    """
    net: resnet18, alexnet, resnet34, inception_v3
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    preprocess = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(), # Operated on original image, rewrite on previous transform.
        transforms.Normalize(norm_mean, norm_std)])

    print('Loading data...')
    testset = ClsDataset(sampling_file, datapath, preprocess)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    net = getattr(models, net)(pretrained=False, num_classes=num_classes)
    modelpath = get_modelpath('prediction_model_92.pkl')
    print('Loading model...', modelpath)
    if len(gpu) > 1:
        net = torch.nn.DataParallel(net).cuda()
        net.load_state_dict(torch.load(modelpath)) # load the finetune weight parameters
    else:
        net = net.cuda()
        net.load_state_dict({k.replace('module.',''):v for k,v in torch.load(modelpath).items()})

    scores_patch_val, predictions_patch_val, namelist_patch_val = net_prediction_oneshop(testloader, net, num_classes)

    save_temp_excel_oneshop(namelist_patch_val, scores_patch_val[:, 1], predictions_patch_val,
                            result_path, num_classes, 'patch', 'test')

    filename = os.path.basename(sampling_file)
    patch_filename = filename.replace('.txt', '_patch.npz')
    savename_patch = os.path.join(result_path, patch_filename)
    np.savez(savename_patch, key_score=scores_patch_val, key_binpred=predictions_patch_val,
             key_namelist=namelist_patch_val)

    scores_patient_val, predictions_patient_val, namelist_patient_val = patient_res_m3_oneshop(scores_patch_val, namelist_patch_val, num_classes)
    
    save_temp_excel_oneshop(namelist_patient_val, scores_patient_val[:, 1], predictions_patient_val, 
                            result_path, num_classes, 'patient', 'test')
    
    patient_filename = filename.replace('.txt', '_patient.npz')
    savename_patient = os.path.join(result_path, patient_filename)
    np.savez(savename_patient,  key_score=scores_patient_val, key_binpred=predictions_patient_val,
             key_namelist=namelist_patient_val)
