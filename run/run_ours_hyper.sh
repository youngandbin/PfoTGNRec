# Define hyperparameter ranges
lambda_mvs=(0.1 0.3 0.5 0.7 0.9)
batch_sizes=(128)
drop_outs=(0.2)
n_heads=(2)
memory_dims=(64)

# Iterate over each combination of hyperparameters
for lambda_mv in "${lambda_mvs[@]}"; do
  for bs in "${batch_sizes[@]}"; do
    for drop_out in "${drop_outs[@]}"; do
      for n_head in "${n_heads[@]}"; do
        for memory_dim in "${memory_dims[@]}"; do
        # Get current time for each iteration
        current_time=$(TZ="Asia/Seoul" date -d "+0 hours" "+%y%m%d_%H%M%S")

        # Execute the python script with the current set of hyperparameters
        python main.py \
            --prefix $current_time \
            --model_name 'ours' \
            --gpu 0 \
            --lambda_mv $lambda_mv \
            --bs $bs \
            --drop_out $drop_out \
            --n_heads $n_head \
            --memory_dim $memory_dim 
        done
      done
    done
  done
done
