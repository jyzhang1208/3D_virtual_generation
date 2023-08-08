TASK_LIST="pick_up_cup
pick_and_lift"

for task in $TASK_LIST; do
    echo $task
    for i in {0..124};do
        echo $i
        python preprocess_data.py --exp_cfg_path configs/all.yaml \
                            --device 0 \
                            --save_episode $i \
                            --task_name $task
    done
done
