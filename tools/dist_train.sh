CUDA_VISIBLE_DEVICES="0,1" 
mpirun --allow-run-as-root -np 2 python3.7 -m jittor.test.test_resnet
# mpirun -np 2 python pix2pix.py --output_path ./results/multi_gpu --batch_size 128 --data_path path_to_your_data