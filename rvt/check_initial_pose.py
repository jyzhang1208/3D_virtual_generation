import os
import pickle
import clip

import torch
import numpy as np
from transformers import (
    CLIPProcessor,
    CLIPTokenizer,
    CLIPTextModel,
    CLIPModel,
)
from PIL import Image
SAVE_PATH = "/media/zjy/e3a9400e-e022-4ed0-b57e-2a86d6ee8488/zjy/RVT/RVT/rvt/data/preprocess_add_lang/"
DATA_PATH = "/media/zjy/e3a9400e-e022-4ed0-b57e-2a86d6ee8488/zjy/RVT/RVT/rvt/data/preprocess_new/"
FORMER_PATH = "/media/zjy/e3a9400e-e022-4ed0-b57e-2a86d6ee8488/zjy/RVT/RVT/rvt/data/train/"
FORMER2_PATH = "/media/zjy/e3a9400e-e022-4ed0-b57e-2a86d6ee8488/zjy/RVT/RVT/rvt/data/train/reach_and_drag/all_variations/episodes/episode0"
EPISODE_FOLDER = 'episode%d'

RLBENCH_TASKS = [
    # "reach_and_drag",
    # "reach_target",
    # "slide_block_to_color_target",
    # "open_drawer",
    # "put_groceries_in_cupboard",
    # "place_shape_in_shape_sorter",
    # "push_buttons",
    # "close_jar",
    # "place_wine_at_rack_location",
    # "light_bulb_in",
    # "insert_onto_square_peg",
    # "meat_off_grill",
    # "stack_cups",
    # "put_item_in_drawer",
    # "turn_tap",
    # "sweep_to_dustpan_of_size",
    # "put_money_in_safe",
    # "stack_blocks",
    # "place_cups",

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

def _clip_encode_text(clip_model, text):
    x = clip_model.token_embedding(text).type(
        clip_model.dtype
    )  # [batch_size, n_ctx, d_model]

    x = x + clip_model.positional_embedding.type(clip_model.dtype)
    x = x.permute(1, 0, 2)  # NLD -> LND
    x = clip_model.transformer(x)
    x = x.permute(1, 0, 2)  # LND -> NLD
    x = clip_model.ln_final(x).type(clip_model.dtype)

    emb = x.clone()
    x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ clip_model.text_projection

    return x, emb

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

for task in RLBENCH_TASKS:
    print(task)
    try:
        clip_model, _ = clip.load("RN50", device="cpu")  # CLIP-ResNet50
        clip_model = clip_model.to("cpu")
        clip_model.eval()
    except RuntimeError:
        print("WARNING: Setting Clip to None. Will not work if replay not on disk.")
        clip_model = None
    # f = open(FORMER_PATH + task + '_all.pkl', 'rb')
    ff = open(DATA_PATH + task + '_all.pkl', 'rb')
    raw_data = pickle.load(ff)
    lang_encoder, lang_tokenizer = load_lang_encoders()
    for i in range(125):
        print(i)
        f = open(FORMER_PATH + task + '/all_variations/episodes/' + f'episode{i}' + '/variation_descriptions.pkl', 'rb')
        desc_data = pickle.load(f)
        temp = open(FORMER_PATH + '5_tasks_variation.pkl', 'rb')
        variation = pickle.load(temp)
        # import pdb;pdb.set_trace()
        ep_var = variation[task]
        lang_total = []
        new_feat = []
        origin_lang_feat = []
        # lang_encoder, lang_tokenizer = load_lang_encoders()
        for sub_descript in desc_data:
            lang_temp = get_lang_token(lang_encoder, lang_tokenizer, sub_descript, 'cuda:0')
            lang_origin = lang_temp[0].cpu().numpy()
            origin_lang_feat.append(lang_origin[0])
            tokens = clip.tokenize([sub_descript]).numpy()
            token_tensor = torch.from_numpy(tokens).to("cpu")
            with torch.no_grad():
                lang_feats, lang_embs = _clip_encode_text(clip_model, token_tensor)
            lang_emb_77 = lang_embs[0].float().detach().cpu().numpy()
            lang_feat_add = lang_feats[0].float().detach().cpu().numpy()
            # import pdb;pdb.set_trace()
            lang_total.append(lang_emb_77)
            new_feat.append(lang_feat_add)
        for j in range(len(ep_var)):
            if ep_var[str(j)]['lang_goal'] == raw_data[i]['lang_goal']:
                raw_data[i]['variation'] = j
                break
        raw_data[i]['lang_emb_77'] = np.array(lang_total)
        raw_data[i]['lang_goal'] = desc_data
        raw_data[i]['new_lang_feat'] = np.array(new_feat)
        raw_data[i]['lang_emb'] = np.array(origin_lang_feat)

    # import pdb;pdb.set_trace()
    # for i in range(len(data)):
    #     print(i)
        # data[i]["poses"][0] = [2.78476566e-01, -8.16252083e-03, 1.47195959e+00, 2.96202711e-06,
        #                        9.92665350e-01, -1.05953472e-06, 1.20895214e-01, 1.00000000e+00]
    save_path = SAVE_PATH + task + '_all.pkl'
    # import pdb;pdb.set_trace()
    with open(save_path, "wb") as f:
        pickle.dump(raw_data, f)
    print("finish!")
