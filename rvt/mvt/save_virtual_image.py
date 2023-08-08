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
FOLDER_PATH = "/media/zjy/e3a9400e-e022-4ed0-b57e-2a86d6ee8488/zjy/RVT/RVT/rvt/data/preprocess_new"
EPISODE_FOLDER = 'episode%d'
DATA_PATH="/media/zjy/e3a9400e-e022-4ed0-b57e-2a86d6ee8488/zjy/RVT/RVT/rvt/data/preprocess_new/"
SAVE_PATH="/media/zjy/e3a9400e-e022-4ed0-b57e-2a86d6ee8488/zjy/RVT/RVT/rvt/data/processed_data_eval/"
# keypoint_num = 0

def load_lang_encoders():
    lang_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    lang_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
    return lang_encoder, lang_tokenizer

def get_lang_token(lang_encoder, lang_tokenizer, lang_goal, device):
    tokens = lang_tokenizer(
        lang_goal, padding=True, return_tensors="pt"
    )
    output = lang_encoder(**tokens)
    prompt = output.pooler_output[None, :, :]  # [1, 512]
    prompt = prompt.detach().to(device=device)
    return prompt

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

def save_virtual_add_eval(img, episode_num, task, frame_num, lang_emb, proprio, lang_goal, action_set, pose_set, terminal_set, reward_set, frame_list):
    if episode_num == 0:
        f = open(DATA_PATH + task + '_all.pkl', 'rb')
        data = pickle.load(f)
    else:
        f = open(SAVE_PATH + task + '_all.pkl', 'rb')
        data = pickle.load(f)

    import torchvision.transforms as T
    virtual_images = list(img)
    lang_embedding = list(lang_emb)
    proprio_feat = list(proprio)
    # pose_set[0] = pose_set[1] * 0
    for i in range(len(action_set)):
        action_set[i] = list(action_set[i])
        pose_set[i] = list(pose_set[i])
    pose_set[0] = [2.78476566e-01, -8.16252083e-03, 1.47195959e+00, 2.96202711e-06,
                   9.92665350e-01, -1.05953472e-06, 1.20895214e-01, 1.00000000e+00]
    lang_target = lang_goal[0][0]
    terminal_set = np.array(terminal_set)
    reward_set = np.array(reward_set)
    frame_list = np.array(frame_list)
    lang_encoder, lang_tokenizer = load_lang_encoders()
    lang_temp = get_lang_token(lang_encoder, lang_tokenizer, lang_target, 'cuda:0')
    lang_new = list(lang_temp[0].cpu().numpy())
    lang_embedding = np.array([lang_embedding[0].tolist()])

    print(len(data))
    print(lang_embedding.shape)
    print(np.array(pose_set))


    save_dict = {
        "task_name": task,
        "episode": episode_num+len(data),
        "timesteps": frame_list,
        "imgs": np.array(virtual_images),
        "lang_emb_77": lang_embedding,
        "lang_emb": np.array(lang_new),
        "proprio": np.array(proprio_feat),
        "lang_goal": lang_target,
        "actions": np.array(action_set),
        "poses": np.array(pose_set),
        "terminals": terminal_set,
        "rewards": reward_set,

    }

    data.append(save_dict)
    # import pdb;pdb.set_trace()
    save_path = SAVE_PATH + task + '_all.pkl'
    with open(save_path, "wb") as f:
        pickle.dump(data, f)



def save_virtual(img, episode_num, task, frame_num, lang_emb, proprio, lang_goal, action_set, pose_set, terminal_set, reward_set, frame_list):
    TASK_NAME = task
    num = episode_num
    """
    direct = os.path.join(FOLDER_PATH, TASK_NAME, EPISODE_FOLDER % num)
    sub_direct1 = os.path.join(FOLDER_PATH, TASK_NAME, EPISODE_FOLDER % num, "viewpoint_1")
    sub_direct2 = os.path.join(FOLDER_PATH, TASK_NAME, EPISODE_FOLDER % num, "viewpoint_2")
    sub_direct3 = os.path.join(FOLDER_PATH, TASK_NAME, EPISODE_FOLDER % num, "viewpoint_3")
    sub_direct4 = os.path.join(FOLDER_PATH, TASK_NAME, EPISODE_FOLDER % num, "viewpoint_4")
    sub_direct5 = os.path.join(FOLDER_PATH, TASK_NAME, EPISODE_FOLDER % num, "viewpoint_5")"""

    direct = os.path.join(FOLDER_PATH, TASK_NAME + "_all")

    try:
        if not os.path.exists(direct):
            os.makedirs(direct)
            # os.makedirs(sub_direct1)
            # os.makedirs(sub_direct2)
            # os.makedirs(sub_direct3)
            # os.makedirs(sub_direct4)
            # os.makedirs(sub_direct5)
            # print(f"create directory successfully!")
            # print(f"directory already exist")
    except OSError:
        print(f"directory fails to create")

    import torchvision.transforms as T
    """
    virtual_point_cloud = img[0, 0:5, 0:3]
    virtual_RGB = img[0, 0:5, 3:6]
    virtual_depth = img[0, 0:5, 6]
    virtual_pixel_loc = img[0, 0:5, 7:]"""
    virtual_images = list(img)
    lang_embedding = list(lang_emb)
    proprio_feat = list(proprio)
    # pose_set[0] = pose_set[1] * 0
    pose_set[0] = [2.78476566e-01, -8.16252083e-03, 1.47195959e+00, 2.96202711e-06,
                   9.92665350e-01, -1.05953472e-06, 1.20895214e-01, 1.00000000e+00]
    for i in range(len(action_set)):
        action_set[i] = list(action_set[i])
        pose_set[i] = list(pose_set[i])
    lang_target = lang_goal[0][0]
    terminal_set = np.array(terminal_set)
    reward_set = np.array(reward_set)
    frame_list = np.array(frame_list)
    # print("lang_goal", lang_goal)
    # print("action_set", action_set)
    # print("pose_set", pose_set)
    # print("terminal_set", terminal_set)
    # print("cccccc", reward_set)
    # print("length", len(terminal_set))
    lang_encoder, lang_tokenizer = load_lang_encoders()
    lang_temp = get_lang_token(lang_encoder, lang_tokenizer, lang_target, 'cuda:0')
    lang_new = list(lang_temp[0].cpu().numpy())
    pose_set[0] = [2.78476566e-01, -8.16252083e-03, 1.47195959e+00, 2.96202711e-06,
                   9.92665350e-01, -1.05953472e-06, 1.20895214e-01, 1.00000000e+00]


    # print(lang_temp.shape)

    save_dict = {
        "task_name": task,
        "episode": episode_num,
        "timesteps": frame_list,
        "imgs": np.array(virtual_images),
        "lang_emb_77": np.array(lang_embedding),
        "lang_emb": np.array(lang_new),
        "proprio": np.array(proprio_feat),
        "lang_goal": lang_target,
        "actions": np.array(action_set),
        "poses": np.array(pose_set),
        "terminals": terminal_set,
        "rewards": reward_set,

    }
    # print(save_dict)
    # import pdb;
    # pdb.set_trace()



    print(direct + f"/ep{episode_num}.pkl")
    save_path = direct + f"/ep{episode_num}.pkl"
    with open(direct + f"/ep{episode_num}.pkl", "wb") as f:
        pickle.dump(save_dict, f)

    temp = open(save_path, 'rb')
    data = pickle.load(temp)
    if terminal_set[frame_num] == True:
        # import numpy as np
        # print("img", np.array(data['virtual_images']).shape)
        print("lang_emb", np.array(data['lang_emb']).shape)
        print("proprio", np.array(data['proprio']).shape)

    """
    # print("111")
    # print(lang_embedding.shape)
    # print(proprio_feat.shape)
    tmp1 = img[0][0][0:3]
    tmp2 = img[0][1][3:6]
    tmp3 = img[0][2][3:6]
    tmp4 = img[0][3][3:6]
    tmp5 = img[0][4][3:6]
    # print(tmp[5][0])
    transform = T.ToPILImage()
    img1 = transform(tmp1)  # over_rgb, keep the same
    # img1.show()
    img2 = transform(tmp2)
    # img2.show()
    img3 = transform(tmp3)
    # img3.show()
    img4 = transform(tmp4)
    # img4.show()
    img5 = transform(tmp5)
    # img5.show()
    save_path1 = sub_direct1 + '/' + str(frame_num) + '.png'
    # img1.save(save_path1)
    save_path2 = sub_direct2 + '/' + str(frame_num) + '.png'
    # img2.save(save_path2)
    save_path3 = sub_direct3 + '/' + str(frame_num) + '.png'
    # img3.save(save_path3)
    save_path4 = sub_direct4 + '/' + str(frame_num) + '.png'
    # img4.save(save_path4)
    save_path5 = sub_direct5 + '/' + str(frame_num) + '.png'
    # img5.save(save_path5)
    """

    if frame_num == 59:
        import pdb;pdb.set_trace()




