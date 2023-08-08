import pickle
# DATA_PATH="/media/zjy/e3a9400e-e022-4ed0-b57e-2a86d6ee8488/zjy/RLBench2/processed_data/setup_checkers+0/r3m/ep0.pkl"
DATA_PATH="/media/zjy/e3a9400e-e022-4ed0-b57e-2a86d6ee8488/zjy/RVT/RVT/rvt/data/preprocess_new/"
SAVE_PATH="/media/zjy/e3a9400e-e022-4ed0-b57e-2a86d6ee8488/zjy/RVT/RVT/rvt/data/preprocess_new"

RLBENCH_TASKS = [
    # "reach_and_drag",
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
    # "reach_target",
    # "open_door",
    # "open_fridge",
    # "open_grill",
    # "open_microwave",
    # "open_oven",
    # "open_box",
    # "open_window",
    # "plug_charger_in_power_supply",
    # "unplug_charger",
    # "turn_oven_on",
    # "take_lid_off_saucepan",
    "move_hanger",
    "sweep_to_dustpan",
    "turn_tap",
    "insert_onto_square_peg",
    "put_money_in_safe",
    "close_jar",
]

for task in RLBENCH_TASKS:
    print(task)
    total_data = []
    for i in range(125):
        data_path = DATA_PATH + f"{task}_all"
        f = open(data_path + '/ep' + str(i) + '.pkl', 'rb')
        data = pickle.load(f)
        total_data.append(data)
    save_path = SAVE_PATH + f"/{task}_all.pkl"
    with open(save_path, "wb") as f:
        pickle.dump(total_data, f)
# for i in range(19):
#     f = open(DATA_PATH + '/ep' + str(i+81) + '.pkl', 'rb')
#     data = pickle.load(f)
#     total_data.append(data)


# save_path = SAVE_PATH + f"/slide_block_to_color_target_all.pkl"
# with open(save_path, "wb") as f:
#     pickle.dump(total_data, f)

temp = open(save_path, 'rb')
temp_data = pickle.load(temp)
import pdb;pdb.set_trace()