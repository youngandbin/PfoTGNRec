for dataset in period_1 period_2 period_3 period_4 period_5 period_6 period_7
do
    for model in BPR LightGCN SGL
    do
        for k in 1 3 5 
        do
            echo $model $dataset $k
            python evaluation.py \
                --model $model \
                --dataset $dataset \
                --k $k;
        done
    done
done
