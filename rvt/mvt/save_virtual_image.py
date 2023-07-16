import os
import pickle

FOLDER_PATH = "/media/zjy/e3a9400e-e022-4ed0-b57e-2a86d6ee8488/zjy/RVT/RVT/rvt/data/preprocess"
EPISODE_FOLDER = 'episode%d'
# keypoint_num = 0



def save_virtual(img, episode_num, task, frame_num, lang_emb, proprio, lang_goal, action_set, pose_set, terminal_set, reward_set):
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
    virtual_images = img
    lang_embedding = lang_emb
    proprio_feat = proprio
    # print("lang_goal", lang_goal)
    # print("action_set", action_set)
    # print("pose_set", pose_set)
    # print("terminal_set", terminal_set)
    # print("reward_set", reward_set)
    # print("length", len(terminal_set))

    save_dict = {
        "task_name": task,
        "episode": episode_num,
        "virtual_images": virtual_images,
        "lang_emb": lang_embedding,
        "proprio": proprio_feat,
        "lang_goal": lang_goal,
        "action_set": action_set,
        "pose_set": pose_set,
        "terminal_set": terminal_set,
        "reward_set": reward_set,

    }



    print(direct + f"/ep{episode_num}.pkl")
    save_path = direct + f"/ep{episode_num}.pkl"
    with open(direct + f"/ep{episode_num}.pkl", "wb") as f:
        pickle.dump(save_dict, f)

    temp = open(save_path, 'rb')
    data = pickle.load(temp)
    if terminal_set[frame_num] == True:
        import numpy as np
        print("img", np.array(data['virtual_images']).shape)
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




