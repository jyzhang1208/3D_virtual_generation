import pickle
import os

DATA_PATH="/media/zjy/e3a9400e-e022-4ed0-b57e-2a86d6ee8488/zjy/RVT/RVT/rvt/data/train/"
def save_variation(var_num, language, task_name):
    save_path = DATA_PATH + f"{task_name}_variation.pkl"
    save_dict ={
        "lang_goal": language[0][0],
        "variation_num": var_num
    }
    if os.path.exists(save_path):
        temp = open(save_path, 'rb')
        total_list = pickle.load(temp)
        total_list.append(save_dict)
        with open(save_path, "wb") as f:
            pickle.dump(total_list, f)
    else:
        total_list = []
        total_list.append(save_dict)
        with open(save_path, "wb") as f:
            pickle.dump(total_list, f)

    temp = open(save_path, 'rb')
    temp_data = pickle.load(temp)
    print(temp_data)
    print(len(temp_data))
