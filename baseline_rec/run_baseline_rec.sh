for dataset in period_1 period_2 period_3 period_4 period_5 period_6 period_7
do
    for model in BPR LightGCN SGL
    do
        echo $model $dataset
        python main.py \
            --model $model \
            --dataset $dataset;
    done
done