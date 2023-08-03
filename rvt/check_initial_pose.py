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
SAVE_PATH = "/media/zjy/e3a9400e-e022-4ed0-b57e-2a86d6ee8488/zjy/RVT/RVT/rvt/data/processed_baseline_eval/"
DATA_PATH = "/media/zjy/e3a9400e-e022-4ed0-b57e-2a86d6ee8488/zjy/RVT/RVT/rvt/data/preprocess/"
EPISODE_FOLDER = 'episode%d'

RLBENCH_TASKS = [
    # "reach_and_drag",
    # "slide_block_to_color_target",
    # "open_drawer",
    # "put_groceries_in_cupboard",
    # "place_shape_in_shape_sorter",
    # "push_buttons",
    # "close_jar",
    "place_wine_at_rack_location",
    # "light_bulb_in",
    # "insert_onto_square_peg",
    # "meat_off_grill",
    "stack_cups",
    "put_item_in_drawer",
    "turn_tap",
    "sweep_to_dustpan_of_size",
    "put_money_in_safe",
    "stack_blocks",
    "place_cups",
]

for task in RLBENCH_TASKS:
    print(task)
    f = open(SAVE_PATH + task + '_all.pkl', 'rb')
    data = pickle.load(f)
    import pdb;pdb.set_trace()
    for i in range(len(data)):
        print(i)
        # data[i]["poses"][0] = [2.78476566e-01, -8.16252083e-03, 1.47195959e+00, 2.96202711e-06,
        #                        9.92665350e-01, -1.05953472e-06, 1.20895214e-01, 1.00000000e+00]
    save_path = DATA_PATH + task + '_all.pkl'
    # import pdb;pdb.set_trace()
    with open(save_path, "wb") as f:
        pickle.dump(data, f)
    print("finish!")
