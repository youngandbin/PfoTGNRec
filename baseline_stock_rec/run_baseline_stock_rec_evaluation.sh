# MVECF-WMF
for target_year in period_all; do 
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

for target_year in period_all; do 
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

for target_year in period_all; do 
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


        
