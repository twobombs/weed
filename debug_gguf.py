#!/usr/bin/env python3
import struct

with open('Qwen3.5-2B-Q4_K_S.gguf', 'rb') as f:
    # Skip header
    f.read(24)  # magic + version + metadata_count + tensor_count
    
    # Read all metadata entries until we find the string array
    for i in range(20):
        key_len = struct.unpack('<I', f.read(4))[0]
        f.read(4)  # padding
        key = f.read(key_len).decode('utf-8')
        value_type = struct.unpack('<I', f.read(4))[0]
        
        print(f'{i}: key={key}, value_type={value_type}')
        
        if value_type == 8:  # string
            str_len = struct.unpack('<I', f.read(4))[0]
            f.read(4)  # padding
            str_data = f.read(str_len).decode('utf-8')
            print(f'   value={str_data}')
        elif value_type == 9:  # array
            arr_type = struct.unpack('<I', f.read(4))[0]
            arr_len = struct.unpack('<I', f.read(4))[0]
            print(f'   arr_type={arr_type}, arr_len={arr_len}')
            # Read array elements
            for j in range(arr_len):
                sl = struct.unpack('<I', f.read(4))[0]
                f.read(4)  # padding
                s = f.read(sl).decode('utf-8')
                print(f'   str[{j}]={s}')
        elif value_type == 4:  # uint32
            val = struct.unpack('<I', f.read(4))[0]
            print(f'   value={val}')
        elif value_type == 6:  # float32
            val = struct.unpack('<f', f.read(4))[0]
            print(f'   value={val}')
        else:
            f.read(4)  # skip
