# MVECF-WMF
for target_year in period_1 period_2 period_3 period_4 period_5 period_6 period_7; do 
    for model_name in mvecf_wmf; do
        for topk in 1 3 5; do
            python evaluation.py \
                --target_year $target_year \
                --model_name $model_name \
                --topk $topk \
                --numbers 9 ;
        done
    done
done

wait

# WMF

for target_year in period_1 period_2 period_3 period_4 period_5 period_6 period_7; do 
    for model_name in wmf; do
        for topk in 1 3 5; do
            python evaluation.py \
                --target_year $target_year \
                --model_name $model_name \
                --topk $topk \
                --numbers 1 ;
        done
    done
done

wait

# WMF-twophase

for target_year in period_1 period_2 period_3 period_4 period_5 period_6 period_7; do 
    for model_name in twophase_wmf; do
        for topk in 1 3 5; do
            python evaluation.py \
                --target_year $target_year \
                --model_name $model_name \
                --topk $topk \
                --numbers 1 ;
        done
    done
done


        