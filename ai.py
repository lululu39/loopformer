import pandas as pd
import numpy as np

def analyze_ncu_report(csv_file):
    """
    Analyze NCU profiling report to calculate arithmetic intensity
    """
    # Read CSV file and clean column names
    df = pd.read_csv(csv_file)
    df.columns = df.columns.str.strip().str.replace('"', '').str.replace(' ', '_')
    
    # Convert metric values to numeric
    df['Metric_Value'] = pd.to_numeric(df['Metric_Value'], errors='coerce').fillna(0)
    
    # Unit conversion function for memory metrics
    def convert_to_bytes(value, unit):
        if pd.isna(unit) or unit == '' or unit.lower() == 'byte':
            return value
        
        multipliers = {
            'kbyte': 1024, 'mbyte': 1024**2, 'gbyte': 1024**3, 'tbyte': 1024**4,
            'kb': 1024, 'mb': 1024**2, 'gb': 1024**3, 'tb': 1024**4
        }
        return value * multipliers.get(unit.lower(), 1)
    
    # Apply unit conversion for memory metrics
    memory_metrics = ['dram__bytes_read.sum', 'dram__bytes_write.sum']
    if 'Metric_Unit' in df.columns:
        df['Converted_Value'] = df.apply(
            lambda row: convert_to_bytes(row['Metric_Value'], row['Metric_Unit']) 
            if row['Metric_Name'] in memory_metrics else row['Metric_Value'], 
            axis=1
        )
    else:
        print("Warning: Metric_Unit column not found. Assuming all values are in bytes.")
        df['Converted_Value'] = df['Metric_Value']
    
    # Group and aggregate metrics
    grouped = df.groupby(['Kernel_Name', 'Metric_Name'])['Converted_Value'].sum().reset_index()
    
    # Calculate total FLOPs
    flop_metrics = [
        'sm__ops_path_tensor_src_bf16_dst_fp32.sum',
    ]
    
    total_flops = 0
    for metric in flop_metrics:
        flops = grouped[grouped['Metric_Name'] == metric]['Converted_Value'].sum()
        total_flops += flops
        if flops > 0:
            print(f"{metric}: {flops:,.0f}")
    
    # Calculate total memory bytes
    memory_read = grouped[grouped['Metric_Name'] == 'dram__bytes_read.sum']['Converted_Value'].sum()
    memory_write = grouped[grouped['Metric_Name'] == 'dram__bytes_write.sum']['Converted_Value'].sum()
    total_memory = memory_read + memory_write
    
    # Results
    print(f"\nMemory Statistics:")
    print(f"  DRAM read: {memory_read:,.0f} bytes ({memory_read/1e9:.2f} GB)")
    print(f"  DRAM write: {memory_write:,.0f} bytes ({memory_write/1e9:.2f} GB)")
    print(f"  Total: {total_memory:,.0f} bytes ({total_memory/1e9:.2f} GB)")
    
    print(f"\nCompute Statistics:")
    print(f"  Total FLOPs: {total_flops:,.0f} ({total_flops/1e12:.2f} TFLOPs)")
    
    if total_memory > 0:
        arithmetic_intensity = total_flops / total_memory
        print(f"  Arithmetic Intensity: {arithmetic_intensity:.2f} FLOPs/byte")
    else:
        arithmetic_intensity = 0
        print(f"  Warning: Total memory is zero, cannot calculate arithmetic intensity")
    
    # Top kernels by FLOPs
    print(f"\nTop 5 kernels by FLOPs:")
    kernel_flops = {}
    for metric in flop_metrics:
        kernel_data = grouped[grouped['Metric_Name'] == metric]
        for _, row in kernel_data.iterrows():
            kernel = row['Kernel_Name']
            kernel_flops[kernel] = kernel_flops.get(kernel, 0) + row['Converted_Value']
    
    for i, (kernel, flops) in enumerate(sorted(kernel_flops.items(), key=lambda x: x[1], reverse=True)[:5]):
        print(f"  {i+1}. {kernel[:60]}... : {flops:,.0f} FLOPs")
    
    # Top kernels by memory
    print(f"\nTop 5 kernels by memory usage:")
    kernel_memory = {}
    for metric in memory_metrics:
        kernel_data = grouped[grouped['Metric_Name'] == metric]
        for _, row in kernel_data.iterrows():
            kernel = row['Kernel_Name']
            kernel_memory[kernel] = kernel_memory.get(kernel, 0) + row['Converted_Value']
    
    for i, (kernel, memory) in enumerate(sorted(kernel_memory.items(), key=lambda x: x[1], reverse=True)[:5]):
        print(f"  {i+1}. {kernel[:60]}... : {memory:,.0f} bytes")
    
    return {
        'total_flops': total_flops,
        'total_memory': total_memory,
        'arithmetic_intensity': arithmetic_intensity,
        'memory_read': memory_read,
        'memory_write': memory_write
    }

def main():
    csv_file = 'report.csv'
    
    print("Analyzing NCU profiling report...")
    print("=" * 60)
    
    results = analyze_ncu_report(csv_file)
    
    print("=" * 60)
    print("Summary:")
    print(f"  Total FLOPs: {results['total_flops']:,.0f}")
    print(f"  Total Memory: {results['total_memory']:,.0f} bytes")
    print(f"  Arithmetic Intensity: {results['arithmetic_intensity']:.2f} FLOPs/byte")

if __name__ == "__main__":
    main()