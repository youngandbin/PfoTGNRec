# Define hyperparameter ranges
lambda_mvs=(0.5)
batch_sizes=(128 256 512)
drop_outs=(0)
n_headss=(2)

# Iterate over each combination of hyperparameters
for lambda_mv in "${lambda_mvs[@]}"; do
  for bs in "${batch_sizes[@]}"; do
    for drop_out in "${drop_outs[@]}"; do
      for n_heads in "${n_headss[@]}"; do
        # Get current time for each iteration
        current_time=$(TZ="Asia/Seoul" date -d "+0 hours" "+%y%m%d_%H%M%S")
        
        # Execute the python script with the current set of hyperparameters
        python main.py \
        --prefix $current_time \
        --model_name 'jodie' \
        --gpu 0 \
        --lambda_mv $lambda_mv \
        --bs $bs \
        --drop_out $drop_out \
        --n_heads $n_heads 
      done
    done
  done
done
