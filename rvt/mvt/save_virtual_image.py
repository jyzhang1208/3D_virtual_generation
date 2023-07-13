import os

FOLDER_PATH = "/media/zjy/e3a9400e-e022-4ed0-b57e-2a86d6ee8488/zjy/RVT/RVT/rvt/data"
EPISODE_FOLDER = 'episode%d'
keypoint_num = 0

def save_virtual(img1, img2, img3, img4, img5, episode_num, task):
    TASK_NAME = task
    num = episode_num
    print(num)
    direct = os.path.join(FOLDER_PATH, TASK_NAME, EPISODE_FOLDER % num)
    sub_direct1 = os.path.join(FOLDER_PATH, TASK_NAME, EPISODE_FOLDER % num, "viewpoint_1")
    sub_direct2 = os.path.join(FOLDER_PATH, TASK_NAME, EPISODE_FOLDER % num, "viewpoint_2")
    sub_direct3 = os.path.join(FOLDER_PATH, TASK_NAME, EPISODE_FOLDER % num, "viewpoint_3")
    sub_direct4 = os.path.join(FOLDER_PATH, TASK_NAME, EPISODE_FOLDER % num, "viewpoint_4")
    sub_direct5 = os.path.join(FOLDER_PATH, TASK_NAME, EPISODE_FOLDER % num, "viewpoint_5")
    try:
        if not os.path.exists(direct):
            os.makedirs(direct)
            os.makedirs(sub_direct1)
            os.makedirs(sub_direct2)
            os.makedirs(sub_direct3)
            os.makedirs(sub_direct4)
            os.makedirs(sub_direct5)
            print(f"create directory successfully!")
        else:
            print(f"directory already exist")
    except OSError:
        print(f"directory fails to create")

    save_path1 = sub_direct1 + str(keypoint_num) + 'png'
    img1.save(save_path1)




