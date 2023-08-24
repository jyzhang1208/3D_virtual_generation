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
SAVE_PATH = "/media/zjy/e3a9400e-e022-4ed0-b57e-2a86d6ee8488/zjy/RVT/RVT/rvt/data/train/"
DATA_PATH = "/media/zjy/e3a9400e-e022-4ed0-b57e-2a86d6ee8488/zjy/RVT/RVT/rvt/data/train/"
PATH_77 = "/media/zjy/e3a9400e-e022-4ed0-b57e-2a86d6ee8488/zjy/RVT/RVT/rvt/data/preprocess_new/"
EPISODE_FOLDER = 'episode%d'

RLBENCH_TASKS = [
    # "move_hanger",
    # "take_lid_off_saucepan",
    # "sweep_to_dustpan",
    # "lamp_on",
    # "lamp_off",
    # "take_money_out_safe",
    # "reach_and_drag",
    # "pick_up_cup",
    # "pick_and_lift",
    # "reach_target",
    # "open_door",
    # "open_fridge",
    # "open_grill",
    # "open_oven",
    # "open_box",
    # "open_window",
    # "plug_charger_in_power_supply",
    # "unplug_charger",
    # "turn_oven_on",
    # "close_jar",
    # "insert_onto_square_peg",
    # "meat_off_grill",
    # "put_money_in_safe",
    # "turn_tap",
    "wipe_desk",
    "take_umbrella_out_of_umbrella_stand",
    "put_books_on_bookshelf",
    "basketball_in_hoop",
    "push_button",
]

total_dict = {}
for task in RLBENCH_TASKS:
    print(task)
    f = open(DATA_PATH + task + '_variation.pkl', 'rb')
    data = pickle.load(f)
    ff = open(PATH_77 + task + '_all.pkl', 'rb')
    data_77 = pickle.load(ff)
    task_dict = {}
    for i in range(len(data)):
        print(i)
        temp_num = data[i]["variation_num"]
        if str(temp_num) not in task_dict:
            task_dict[str(temp_num)] = {}
            task_dict[str(temp_num)]["lang_goal"] = data[i]["lang_goal"]
            for j in range(len(data_77)):
                if data_77[j]["lang_goal"] == data[i]["lang_goal"]:
                    task_dict[str(temp_num)]["lang_emb_77"] = data_77[j]["lang_emb_77"]
                    break
                # else:
                #     import pdb;pdb.set_trace()

    total_dict[task] = task_dict
    # import pdb;pdb.set_trace()

save_path = SAVE_PATH + f"/5_tasks_variation.pkl"
with open(save_path, "wb") as f:
    pickle.dump(total_dict, f)
print("successfully save!")
ff = open(SAVE_PATH + f"/5_tasks_variation.pkl", 'rb')
data = pickle.load(ff)
import pdb;pdb.set_trace()
