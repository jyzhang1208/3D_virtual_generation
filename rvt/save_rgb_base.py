import os
import pickle

import torch
import numpy as np
from transformers import (
    CLIPProcessor,
    CLIPTokenizer,
    CLIPTextModel,
    CLIPModel,
)
from PIL import Image
SAVE_PATH = "/media/zjy/e3a9400e-e022-4ed0-b57e-2a86d6ee8488/zjy/RVT/RVT/rvt/data/processed_baseline_eval"
DATA_PATH = "/media/zjy/e3a9400e-e022-4ed0-b57e-2a86d6ee8488/zjy/RVT/RVT/rvt/data/processed_data_baseline"
FORMER_PATH = "/media/zjy/e3a9400e-e022-4ed0-b57e-2a86d6ee8488/zjy/RVT/RVT/rvt/data/processed_data_eval"
# DATA_PATH = "/media/zjy/e3a9400e-e022-4ed0-b57e-2a86d6ee8488/zjy/RVT/RVT/rvt/data/preprocess"
# SAVE_PATH = "/media/zjy/e3a9400e-e022-4ed0-b57e-2a86d6ee8488/zjy/RVT/RVT/rvt/data/processed_data_baseline"
EPISODE_FOLDER = 'episode%d'

def get_r3m_features(img_step, model):
    import torchvision.transforms as T

    ## DEFINE PREPROCESSING
    transforms = T.Compose(
        [T.Resize(256), T.CenterCrop(224), T.ToTensor()]
    )  # ToTensor() divides by 255

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    img_step_features = []
    for img in img_step:
        preprocessed_image = transforms(Image.fromarray(img.astype(np.uint8))).reshape(
            -1, 3, 224, 224
        )
        preprocessed_image.to(device)
        with torch.no_grad():
            embedding = (
                model(preprocessed_image * 255.0).detach().cpu().numpy()
            )  ## R3M expects image input to be [0-255]
            # print(embedding.shape) # [1, 2048]
            img_step_features.append(embedding)
    img_step_features = np.concatenate(img_step_features, axis=1).squeeze()
    return img_step_features
def save_rgb(base_rgb_list, task_name, episode_idx):
    from r3m import load_r3m
    model = load_r3m("resnet50")
    model.eval()
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    model.to(device)

    base_rgb_list = np.array(base_rgb_list)
    # base_rgb_list = torch.from_numpy(base_rgb_list)

    img_embed_ls = []
    for i in range(len(base_rgb_list)):
        img_step = base_rgb_list[i]
        img_step_features = get_r3m_features(img_step, model)
        img_embed_ls.append(img_step_features)

    print("img_embed_ls", img_embed_ls)
    if episode_idx == 0:
        data_path = DATA_PATH + f"/{task_name[0]}_all.pkl"
        temp = open(data_path, 'rb')
        data = pickle.load(temp)
    else:
        save_path = SAVE_PATH + f"/{task_name[0]}_all.pkl"
        temp = open(save_path, 'rb')
        data = pickle.load(temp)

    TD_path = FORMER_PATH + f"/{task_name[0]}_all.pkl"
    TD_temp = open(TD_path, 'rb')
    TD_data = pickle.load(TD_temp)
    total_save = data

    for i in range(25):
        temp_data = TD_data[i+99]
        # temp_data = data[i]
        temp_data["poses"][0] = [ 2.78476566e-01, -8.16252083e-03, 1.47195959e+00, 2.96202711e-06,
                                9.92665350e-01, -1.05953472e-06, 1.20895214e-01, 1.00000000e+00]
        if i == episode_idx :
            save_dict = {
                "task_name": temp_data["task_name"],
                "episode": temp_data["episode"],
                "timesteps": temp_data["timesteps"],
                "imgs": np.array(img_embed_ls),
                "lang_emb_77": temp_data["lang_emb_77"],
                "lang_emb": temp_data["lang_emb"],
                "proprio": temp_data["proprio"],
                "lang_goal": temp_data["lang_goal"],
                "actions": temp_data["actions"],
                "poses": temp_data["poses"],
                "terminals": temp_data["terminals"],
                "rewards": temp_data["rewards"],
            }
            # if i < 86:
            #     total_save[i] = save_dict
            # else:
            #     total_save[episode_idx - 1] = save_dict
            total_save.append(save_dict)
            break

    # if episode_idx == 99:
    save_path = SAVE_PATH + f"/{task_name[0]}_all.pkl"
    with open(save_path, "wb") as f:
        pickle.dump(total_save, f)
    print("successfully save!")

    # tmp = open(save_path, 'rb')
    # datatmp = pickle.load(tmp)
        # print(datatmp[])


