# MVECF-WMF

for target_year in period_1 period_2 period_3 period_4 period_5 period_6 period_7; do 
    num=0
    for gamma in 1 3 5; do
        for reg_param_mv in 0.1 1 10; do
            num=$((num + 1))
            python script_mvecf_wmf.py \
                --target_year $target_year \
                --gamma $gamma \
                --reg_param_mv $reg_param_mv \
                --number $num \
                --epochs 20 ;
        done
    done
done

wait

 # WMF

for target_year in period_1 period_2 period_3 period_4 period_5 period_6 period_7; do 
    num=0
    num=$((num + 1))
    python script_wmf.py \
        --target_year $target_year \
        --number $num \
        --epochs 20 ;
done

wait

# WMF-twophase

for target_year in period_1 period_2 period_3 period_4 period_5 period_6 period_7; do 
    num=0
    num=$((num + 1))
    python script_twophase.py \
        --target_year $target_year \
        --number $num ;
done
        
        