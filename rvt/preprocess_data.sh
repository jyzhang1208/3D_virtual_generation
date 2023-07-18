for i in {60..99}; do
    echo $i
    python preprocess_data.py --exp_cfg_path=configs/all.yaml \
                        --device=0 \
                        --save_episode=$i \

done