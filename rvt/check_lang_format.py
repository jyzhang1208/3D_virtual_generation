import pickle
from transformers import (
    CLIPProcessor,
    CLIPTokenizer,
    CLIPTextModel,
    CLIPModel,
)
import numpy as np

# DATA_PATH="/media/zjy/e3a9400e-e022-4ed0-b57e-2a86d6ee8488/zjy/RLBench2/processed_data/setup_checkers+0/r3m/ep0.pkl"
DATA_PATH="/media/zjy/e3a9400e-e022-4ed0-b57e-2a86d6ee8488/zjy/RVT/RVT/rvt/data/processed_data_baseline/"
SAVE_PATH="/media/zjy/e3a9400e-e022-4ed0-b57e-2a86d6ee8488/zjy/RVT/RVT/rvt/data/processed_data_baseline/"

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

# total_data = []
for task in RLBENCH_TASKS:
    print(task)
    f = open(DATA_PATH + task + '_all.pkl', 'rb')
    data = pickle.load(f)
    # import pdb;pdb.set_trace()
    lang_encoder, lang_tokenizer = load_lang_encoders()
    for i in range(len(data)):
        print(i)
        raw_lang = data[i]["lang_goal"]
        lang_temp = get_lang_token(lang_encoder, lang_tokenizer, raw_lang, 'cuda:0')
        lang_new = list(lang_temp[0].cpu().numpy())
        data[i]["lang_emb"] = np.array(lang_new)
        data[i]["lang_emb_77"] = np.array([data[i]["lang_emb_77"][0].tolist()])
    save_path = SAVE_PATH + task + '_all.pkl'
    with open(save_path, "wb") as f:
        pickle.dump(data, f)
        # import pdb;pdb.set_trace()
# for i in range(19):
#     f = open(DATA_PATH + '/ep' + str(i+81) + '.pkl', 'rb')
#     data = pickle.load(f)
#     total_data.append(data)


# save_path = SAVE_PATH + f"/place_cups_all.pkl"
# with open(save_path, "wb") as f:
#     pickle.dump(total_data, f)

# temp = open(save_path, 'rb')
# temp_data = pickle.load(temp)
import pdb;pdb.set_trace()