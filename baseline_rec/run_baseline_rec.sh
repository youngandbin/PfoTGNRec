for dataset in period_all
do
    for model in BPR LightGCN SGL
    do
        echo $model $dataset
        python main.py \
            --model $model \
            --dataset $dataset;
    done
done
