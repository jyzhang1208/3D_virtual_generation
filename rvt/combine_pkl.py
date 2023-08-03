import pickle
# DATA_PATH="/media/zjy/e3a9400e-e022-4ed0-b57e-2a86d6ee8488/zjy/RLBench2/processed_data/setup_checkers+0/r3m/ep0.pkl"
DATA_PATH="/media/zjy/e3a9400e-e022-4ed0-b57e-2a86d6ee8488/zjy/RVT/RVT/rvt/data/preprocess/slide_block_to_color_target_all"
SAVE_PATH="/media/zjy/e3a9400e-e022-4ed0-b57e-2a86d6ee8488/zjy/RVT/RVT/rvt/data/preprocess"

total_data = []
for i in range(100):
    f = open(DATA_PATH + '/ep' + str(i) + '.pkl', 'rb')
    data = pickle.load(f)
    total_data.append(data)
# for i in range(19):
#     f = open(DATA_PATH + '/ep' + str(i+81) + '.pkl', 'rb')
#     data = pickle.load(f)
#     total_data.append(data)


save_path = SAVE_PATH + f"/slide_block_to_color_target_all.pkl"
with open(save_path, "wb") as f:
    pickle.dump(total_data, f)

temp = open(save_path, 'rb')
temp_data = pickle.load(temp)
import pdb;pdb.set_trace()