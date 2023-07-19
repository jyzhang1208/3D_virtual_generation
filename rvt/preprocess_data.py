import os
import time
import tqdm
import random
import yaml
import argparse
import sys
sys.path.append('..')

from collections import defaultdict
from contextlib import redirect_stdout
import torchvision.transforms as T

import torch
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import config as exp_cfg_mod
import rvt.models.rvt_agent as rvt_agent
import rvt.utils.ddp_utils as ddp_utils
import mvt.config as mvt_cfg_mod

from mvt import MVT
from rvt.models.rvt_agent import print_eval_log, print_loss_log
from rvt.utils.get_dataset import get_dataset
from rvt.utils.rvt_utils import (
    TensorboardManager,
    short_name,
    get_num_feat,
    load_agent,
    RLBENCH_TASKS,
)
from rvt.utils.peract_utils import (
    CAMERAS,
    SCENE_BOUNDS,
    IMAGE_SIZE,
    DATA_FOLDER,
)
from save_rgb_base import save_rgb
from PIL import Image
import cv2


# new train takes the dataset as input
def train(agent, dataset, training_iterations, rank=0, base_rgb=0):
    agent.train()
    log = defaultdict(list)

    data_iter = iter(dataset)
    iter_command = range(training_iterations)

    for iteration in tqdm.tqdm(
        iter_command, disable=(rank != 0), position=0, leave=True
    ):

        raw_batch = next(data_iter)
        batch = {
            k: v.to(agent._device)
            for k, v in raw_batch.items()
            if type(v) == torch.Tensor
        }
        # import pdb;
        # pdb.set_trace()
        batch["tasks"] = raw_batch["tasks"]
        batch["lang_goal"] = raw_batch["lang_goal"]
        update_args = {
            "step": iteration,
        }
        # format: batch_size, 1, channel, pixel, pixel

        # print(raw_batch['right_shoulder_rgb'].shape)
        # print(raw_batch['right_shoulder_rgb'][0][0][1][2])
        right_shoulder_rgb = batch['right_shoulder_rgb'].permute(0, 1, 3, 4, 2).cpu().numpy()
        front_rgb = batch['front_rgb'].permute(0, 1, 3, 4, 2).cpu().numpy()
        left_shoulder_rgb = batch['left_shoulder_rgb'].permute(0, 1, 3, 4, 2).cpu().numpy()
        wrist_rgb = batch['wrist_rgb'].permute(0, 1, 3, 4, 2).cpu().numpy()
        terminal = batch['terminal'].cpu().numpy()
        # print("right_shoulder_rgb", right_shoulder_rgb[0].shape)
        # print("tasks", batch['tasks'])
        right_shoulder_rgb = list(right_shoulder_rgb[0][0])
        front_rgb = list(front_rgb[0][0])
        left_shoulder_rgb = list(left_shoulder_rgb[0][0])
        wrist_rgb = list(wrist_rgb[0][0])
        base_rgb.append([right_shoulder_rgb, front_rgb, left_shoulder_rgb, wrist_rgb])
        # base_rgb.append(front_rgb)
        # base_rgb.append(left_shoulder_rgb)
        # base_rgb.append(wrist_rgb)
        # base_rbg = [base_rgb]
        # import numpy as np
        # print(np.array(base_rgb).shape)
        if terminal[0] == 1:
            save_rgb(base_rgb, batch['tasks'], batch['episode_idx'])

        # import pdb;pdb.set_trace()
        # tmp[0], tmp[1], tmp[2] = tmp[2], tmp[1], tmp[0]
        # tmp = tmp.cpu().numpy()
        # print("333")
        # transform = T.ToPILImage()
        # img = transform(tmp)
        # img.show()
        # print(tmp)
        # tmp = tmp.permute(1, 2, 0)
        # print(tmp[0])
        # print(tmp[0].shape)
        # print("lang_goal", batch["lang_goal"])
        # print("lang_goal_embs", batch["lang_goal_embs"].shape)
        # print("action", batch["action"])
        # print("gripper_pose_tp1", batch["gripper_pose_tp1"])
        # print("reward", batch["reward"])
        # import pdb;pdb.set_trace()


        update_args.update(
            {
                "replay_sample": batch,
                "backprop": True,
                "reset_log": (iteration == 0),
                "eval_log": False,
            }
        )
        agent.update(**update_args)

    if rank == 0:
        log = print_loss_log(agent)

    return log


def save_agent(agent, path, epoch):
    model = agent._network
    optimizer = agent._optimizer
    lr_sched = agent._lr_sched

    if isinstance(model, DDP):
        model_state = model.module.state_dict()
    else:
        model_state = model.state_dict()

    torch.save(
        {
            "epoch": epoch,
            "model_state": model_state,
            "optimizer_state": optimizer.state_dict(),
            "lr_sched_state": lr_sched.state_dict(),
        },
        path,
    )


def get_tasks(exp_cfg):
    parsed_tasks = exp_cfg.tasks.split(",")
    if parsed_tasks[0] == "all":
        tasks = RLBENCH_TASKS
        # import pdb;pdb.set_trace()
        print("1111!", RLBENCH_TASKS)
    else:
        tasks = parsed_tasks
    return tasks


def get_logdir(cmd_args, exp_cfg):
    log_dir = os.path.join(cmd_args.log_dir, exp_cfg.exp_id)
    os.makedirs(log_dir, exist_ok=True)
    return log_dir


def dump_log(exp_cfg, mvt_cfg, cmd_args, log_dir):
    with open(f"{log_dir}/exp_cfg.yaml", "w") as yaml_file:
        with redirect_stdout(yaml_file):
            print(exp_cfg.dump())

    with open(f"{log_dir}/mvt_cfg.yaml", "w") as yaml_file:
        with redirect_stdout(yaml_file):
            print(mvt_cfg.dump())

    args = cmd_args.__dict__
    with open(f"{log_dir}/args.yaml", "w") as yaml_file:
        yaml.dump(args, yaml_file)


def experiment(rank, cmd_args, devices, port):
    """experiment.

    :param rank:
    :param cmd_args:
    :param devices: list or int. if list, we use ddp else not
    """
    device = devices[rank]
    device = f"cuda:{device}"
    ddp = len(devices) > 1
    ddp_utils.setup(rank, world_size=len(devices), port=port)
    save_episode = cmd_args.save_episode

    exp_cfg = exp_cfg_mod.get_cfg_defaults()
    if cmd_args.exp_cfg_path != "":
        exp_cfg.merge_from_file(cmd_args.exp_cfg_path)
    if cmd_args.exp_cfg_opts != "":
        exp_cfg.merge_from_list(cmd_args.exp_cfg_opts.split(" "))

    if ddp:
        print(f"Running DDP on rank {rank}.")

    old_exp_cfg_peract_lr = exp_cfg.peract.lr
    old_exp_cfg_exp_id = exp_cfg.exp_id

    exp_cfg.peract.lr *= len(devices) * exp_cfg.bs
    if cmd_args.exp_cfg_opts != "":
        exp_cfg.exp_id += f"_{short_name(cmd_args.exp_cfg_opts)}"
    if cmd_args.mvt_cfg_opts != "":
        exp_cfg.exp_id += f"_{short_name(cmd_args.mvt_cfg_opts)}"

    if rank == 0:
        print(f"dict(exp_cfg)={dict(exp_cfg)}")
    exp_cfg.freeze()

    # Things to change
    BATCH_SIZE_TRAIN = exp_cfg.bs
    NUM_TRAIN = 100
    # to match peract, iterations per epoch
    TRAINING_ITERATIONS = int(10000 // (exp_cfg.bs * len(devices) / 16))
    EPOCHS = exp_cfg.epochs
    TRAIN_REPLAY_STORAGE_DIR = "replay/replay_train"
    TEST_REPLAY_STORAGE_DIR = "replay/replay_val"
    log_dir = get_logdir(cmd_args, exp_cfg)
    tasks = get_tasks(exp_cfg)
    print("Training on {} tasks: {}".format(len(tasks), tasks))

    t_start = time.time()
    get_dataset_func = lambda: get_dataset(
        tasks,
        BATCH_SIZE_TRAIN,
        None,
        TRAIN_REPLAY_STORAGE_DIR,
        None,
        DATA_FOLDER,
        NUM_TRAIN,
        None,
        cmd_args.refresh_replay,
        device,
        num_workers=exp_cfg.num_workers,
        only_train=True,
        sample_distribution_mode=exp_cfg.sample_distribution_mode,
        save_ep=save_episode,
    )
    train_dataset, _, episode_idx, total_lang, total_action, total_pose, total_terminal, total_reward = get_dataset_func()
    t_end = time.time()
    print("Created Dataset. Time Cost: {} minutes".format((t_end - t_start) / 60.0))
    print(train_dataset)
    base_rgb = []

    if exp_cfg.agent == "our":
        mvt_cfg = mvt_cfg_mod.get_cfg_defaults()
        if cmd_args.mvt_cfg_path != "":
            mvt_cfg.merge_from_file(cmd_args.mvt_cfg_path)
        if cmd_args.mvt_cfg_opts != "":
            mvt_cfg.merge_from_list(cmd_args.mvt_cfg_opts.split(" "))

        mvt_cfg.feat_dim = get_num_feat(exp_cfg.peract)
        mvt_cfg.freeze()

        torch.cuda.set_device(device)
        torch.cuda.empty_cache()
        rvt = MVT(
            renderer_device=device,
            task_name=tasks[0],
            episode_num=episode_idx,
            total_lang = total_lang,
            total_action = total_action,
            total_pose = total_pose,
            total_terminal = total_terminal,
            total_reward = total_reward,
            **mvt_cfg,
        ).to(device) # multi-view transformer
        if ddp:
            rvt = DDP(rvt, device_ids=[device])

        agent = rvt_agent.RVTAgent(
            network=rvt,
            image_resolution=[IMAGE_SIZE, IMAGE_SIZE],
            add_lang=mvt_cfg.add_lang,
            scene_bounds=SCENE_BOUNDS,
            cameras=CAMERAS,
            log_dir=f"{log_dir}/test_run/",
            cos_dec_max_step=EPOCHS * TRAINING_ITERATIONS,
            **exp_cfg.peract,
            **exp_cfg.rvt,
        )
        agent.build(training=True, device=device)
    else:
        assert False, "Incorrect agent"

    start_epoch = 0
    end_epoch = EPOCHS
    if exp_cfg.resume != "":
        agent_path = exp_cfg.resume
        print(f"Recovering model and checkpoint from {exp_cfg.resume}")
        epoch = load_agent(agent_path, agent, only_epoch=False)
        start_epoch = epoch + 1
    dist.barrier()

    if rank == 0:
        ## logging unchanged values to reproduce the same setting
        temp1 = exp_cfg.peract.lr
        temp2 = exp_cfg.exp_id
        exp_cfg.defrost()
        exp_cfg.peract.lr = old_exp_cfg_peract_lr
        exp_cfg.exp_id = old_exp_cfg_exp_id
        dump_log(exp_cfg, mvt_cfg, cmd_args, log_dir)
        exp_cfg.peract.lr = temp1
        exp_cfg.exp_id = temp2
        exp_cfg.freeze()
        tb = TensorboardManager(log_dir)

    print("Start training ...", flush=True)
    i = start_epoch
    while True:
        if i == end_epoch:
            break

        print(f"Rank [{rank}], Epoch [{i}]: Training on train dataset")
        out = train(agent, train_dataset, TRAINING_ITERATIONS, rank, base_rgb)

        if rank == 0:
            tb.update("train", i, out)

        if rank == 0:
            # TODO: add logic to only save some models
            save_agent(agent, f"{log_dir}/model_{i}.pth", i)
            save_agent(agent, f"{log_dir}/model_last.pth", i)
        i += 1

    if rank == 0:
        tb.close()
        print("[Finish]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.set_defaults(entry=lambda cmd_args: parser.print_help())

    parser.add_argument("--refresh_replay", action="store_true", default=False)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--mvt_cfg_path", type=str, default="")
    parser.add_argument("--exp_cfg_path", type=str, default="")

    parser.add_argument("--mvt_cfg_opts", type=str, default="")
    parser.add_argument("--exp_cfg_opts", type=str, default="")

    parser.add_argument("--log-dir", type=str, default="runs")
    parser.add_argument("--save_episode", type=int, default=0)
    parser.add_argument("--with-eval", action="store_true", default=False)

    cmd_args = parser.parse_args()
    del (
        cmd_args.entry
    )  # hack for multi processing -- removes an argument called entry which is not picklable

    devices = cmd_args.device.split(",")
    devices = [int(x) for x in devices]

    port = (random.randint(0, 3000) % 3000) + 27000
    mp.spawn(experiment, args=(cmd_args, devices, port), nprocs=len(devices), join=True)