import pickle
import numpy as np
# DATA_PATH="/media/zjy/e3a9400e-e022-4ed0-b57e-2a86d6ee8488/zjy/RLBench2/processed_data/setup_checkers+0/r3m/ep0.pkl"
DATA_PATH="/media/zjy/e3a9400e-e022-4ed0-b57e-2a86d6ee8488/zjy/RVT/RVT/rvt/data/processed_data_baseline"
# SAVE_PATH="/media/zjy/e3a9400e-e022-4ed0-b57e-2a86d6ee8488/zjy/RVT/RVT/rvt/data/preprocess"

save_path = DATA_PATH + f"/stack_blocks_all.pkl"
# with open(save_path, "wb") as f:
#     pickle.dump(total_save, f)

tmp = open(save_path, 'rb')
datatmp = pickle.load(tmp)
for i in range(len(datatmp)):
    datatmp[i]["imgs"] = np.array(datatmp[i]["imgs"])

with open(save_path, "wb") as f:
    pickle.dump(datatmp, f)
import pdb;pdb.set_trace()
# print(datatmp[])