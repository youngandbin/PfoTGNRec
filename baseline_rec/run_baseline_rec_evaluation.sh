for dataset in period_all
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
