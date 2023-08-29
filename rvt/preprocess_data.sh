TASK_LIST="
push_button
"

for task in $TASK_LIST; do
    echo $task
    for i in {0..3};do
        echo $i
        python preprocess_data.py --exp_cfg_path configs/all.yaml \
                            --device 0 \
                            --save_episode $i \
                            --task_name $task
    done
done
