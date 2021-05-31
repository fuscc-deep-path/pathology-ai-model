import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from tqdm import tqdm


def save_results(namelist, scores, predictions, nCls):
    """
    which_type: patch, patient
    """
    def to_dict_list(json_results):
        keys = json_results.get("name").keys()

        return [{
            "name": json_results.get("name").get(key),
            "score": json_results.get("score").get(key, None),
            "prediction": json_results.get("prediction").get(key, None)
        } for key in keys]

    if nCls==2:
        results = pd.DataFrame({"name": namelist,
                                "score": scores,
                                "prediction": predictions}).to_json()
    elif nCls==4:
        results = pd.DataFrame({"name": namelist,
                                "score": {},
                                "prediction": predictions}).to_json()
    elif nCls==1:
        results = pd.DataFrame({"name": namelist,
                                "score": scores,
                                "prediction": {}}).to_json()

    return to_dict_list(json.loads(results))


def net_prediction_oneshop(dataloader, model, Cls):

    scores = np.empty([0, Cls])
    predictions = np.array([])
    namelist = np.array([])

    model.eval()
    with torch.no_grad():
        for i, (img, name) in tqdm(enumerate(dataloader)):
            out = model(img.cuda())

            prob = F.softmax(out, 1, _stacklevel=5)
            pred = torch.argmax(prob, dim=1)

            predictions = np.concatenate((predictions, pred.cpu().numpy()), axis=0)
            scores = np.concatenate((scores, prob.cpu().numpy()), axis=0)
            namelist = np.concatenate((namelist, name), axis=0)
    return scores, predictions, namelist


def patient_res_m3_oneshop(scores_patch, namelist_patch, Cls):
    scores_patient = np.empty([0, Cls])
    predictions_patient = np.array([])
    namelist_patient = np.array([])
    pid = np.array([name.split('/')[-1].split('_')[0] for name in namelist_patch])

    u, counts = np.unique(pid, return_counts=True)
    for id in u:
        sid_score = scores_patch[pid == id, :]
        sid_score_mean = sid_score.mean(axis=0)

        scores_patient = np.append(scores_patient, sid_score_mean) # axis=0是按照列求和
        scores_patient = np.reshape(scores_patient, (-1, Cls))
        predictions_patient = np.append(predictions_patient, np.where(sid_score_mean == np.max(sid_score_mean)))
        namelist_patient = np.append(namelist_patient, id)

    return scores_patient, predictions_patient, namelist_patient

def get_modelpath(model_name):
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', model_name)
