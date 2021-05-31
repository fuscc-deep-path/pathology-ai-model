import os
import json
import numpy as np
import random
import torch
from torch.utils.data import DataLoader
from torchvision import models, transforms

from pathology_ai_model.data_reader import ClsDataset
from pathology_ai_model.utils import get_modelpath, net_prediction_oneshop, patient_res_m3_oneshop, save_results


def start_model(datapath, sampling_file, root_dir, model_type, seed=2020, gpu="0", net="resnet18",
                num_classes=2, num_workers=4, batch_size=256, norm_mean=[0.8201, 0.5207, 0.7189],
                norm_std=[0.1526, 0.1542, 0.1183]):
    """
    Arguments:
      model_type: 'PIK3CA_Mutation', 'BLIS', 'IM', 'LAR',  'MES'
      net: resnet18, alexnet, resnet34, inception_v3

    Results:
      root_dir: ./FUSCC001_models/
      patch.json: ${root_dir}/${model_type}/patch.json
      patch.npz: ${root_dir}/${model_type}/patch.npz
      patient.json: ${root_dir}/${model_type}/patient.json
      patient.npz: ${root_dir}/${model_type}/patient.npz
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
    modelpath = get_modelpath(model_type + '.pkl')
    print('Loading model...', modelpath)

    if len(gpu) > 1:
        net = torch.nn.DataParallel(net).cuda()
        net.load_state_dict(torch.load(modelpath)) # load the finetune weight parameters
    else:
        net = net.cuda()
        net.load_state_dict({k.replace('module.',''):v for k,v in torch.load(modelpath).items()})

    # Patch Output: patch.json / patch.npz
    scores_patch, predictions_patch, namelist_patch = net_prediction_oneshop(testloader, net, num_classes)

    patch_results = save_results(namelist_patch, scores_patch[:, 1], predictions_patch, num_classes)
    with open(os.path.join(root_dir, model_type, 'patch.json'), 'w') as f:
        json.dump(patch_results, f)

    savename_patch = os.path.join(root_dir, model_type, 'patch.npz')
    np.savez(savename_patch, key_score=scores_patch, key_binpred=predictions_patch, key_namelist=namelist_patch)

    # Patient Output: patient.json / patient.npz
    scores_patient, predictions_patient, namelist_patient = patient_res_m3_oneshop(scores_patch, namelist_patch, num_classes)
    patient_results = save_results(namelist_patient, scores_patient[:, 1], predictions_patient, num_classes)
    with open(os.path.join(root_dir, model_type, 'patient.json'), 'w') as f:
        json.dump(patient_results[0], f)
    
    savename_patient = os.path.join(root_dir, model_type, 'patient.npz')
    np.savez(savename_patient,  key_score=scores_patient, key_binpred=predictions_patient, key_namelist=namelist_patient)

    with open(os.path.join(root_dir, model_type, 'prediction.json'), 'w') as f:
        results = {
          "model": model_type,
          "patient": patient_results[0],
          "patch": patch_results
        }
        json.dump(results, f)
