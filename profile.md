ncu --metrics sm__ops_path_tensor_src_bf16_dst_fp32.sum,dram__bytes_read.sum,dram__bytes_write.sum \
    --target-processes all --replay-mode kernel --export ncu-report-eval --force-overwrite \
    python ncu_bf16_eval.py

ncu --import ncu-report-eval.ncu-rep --csv > report.csv

python ai.py

ncu --metrics sm__ops_path_tensor_src_bf16_dst_fp32.sum,dram__bytes_read.sum,dram__bytes_write.sum \
    --target-processes all --replay-mode kernel --export ncu-report-train --force-overwrite \
    python ncu_bf16_train.py

ncu --import ncu-report-train.ncu-rep --csv > report.csv

python ai.py