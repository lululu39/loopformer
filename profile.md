ncu --metrics sm__ops_path_tensor_src_bf16_dst_fp32.sum,sm__ops_path_tensor_src_bf16_dst_fp32_sparsity_off.sum,sm__ops_path_tensor_src_bf16_dst_fp32_sparsity_on.sum,dram__bytes_read.sum,dram__bytes_write.sum \
    --target-processes all --replay-mode kernel --export ncu-report --force-overwrite \
    python ncu_bf16.py

ncu --import ncu-report.ncu-rep --csv > report.csv

python ai.py