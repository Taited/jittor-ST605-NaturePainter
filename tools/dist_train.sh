export CUDA_VISIBLE_DEVICES=4,5,6,7
mpirun --allow-run-as-root -np 4 python3.7 main.py --output_path ./training_results/flickr_valid