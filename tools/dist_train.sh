export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
mpirun --allow-run-as-root -np 8 python3.7 main.py --output_path ./training_results/flickr_spectralG_syncBN --train_batch_size 24 --valid_batch_size 8