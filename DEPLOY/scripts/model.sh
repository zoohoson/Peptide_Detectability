nohup python ../dbydeep_model.py \
    --retrain-flag True \
    --data-path /data/2021_SJH_detectability/data_cross_species/raw/mouse/KLife/KLife.csv \
    --model-path /home/bis/2021_SJH_detectability/DbyDeep/log/model_DbyDeep_04_False.h5 \
    --save-path ../data/ \
    --job-name DbyDeep_retrained \
     >../../revision/log/result_prediction_gru.out &