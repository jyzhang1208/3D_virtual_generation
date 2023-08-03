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
PATH_77 = "/media/zjy/e3a9400e-e022-4ed0-b57e-2a86d6ee8488/zjy/RVT/RVT/rvt/data/preprocess/"
EPISODE_FOLDER = 'episode%d'

RLBENCH_TASKS = [
    "reach_and_drag",
    "slide_block_to_color_target",
    "open_drawer",
    "put_groceries_in_cupboard",
    "place_shape_in_shape_sorter",
    "push_buttons",
    "close_jar",
    "place_wine_at_rack_location",
    "light_bulb_in",
    "insert_onto_square_peg",
    "meat_off_grill",
    "stack_cups",
    "put_item_in_drawer",
    "turn_tap",
    "sweep_to_dustpan_of_size",
    "put_money_in_safe",
    "stack_blocks",
    "place_cups",
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

save_path = SAVE_PATH + f"/18_tasks_variation.pkl"
with open(save_path, "wb") as f:
    pickle.dump(total_dict, f)
print("successfully save!")
