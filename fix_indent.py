
import os

file_path = "app.py"
print(f"Reading {file_path}")

with open(file_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

new_lines = []
for i, line in enumerate(lines):
    line_num = i + 1
    content = line
    
    # Logic to decide if we indent
    should_indent = False
    
    # Block 1: 850-855
    if 850 <= line_num <= 855:
        should_indent = True
        
    # Block 2: 857-1015
    elif 857 <= line_num <= 1015:
        should_indent = True
        
    # Block 3: 1017-1123
    elif 1017 <= line_num <= 1123:
        should_indent = True
        
    # Block 4: 1125-1231
    elif 1125 <= line_num <= 1231:
        should_indent = True
        
    # Block 5: 1233 to End
    elif 1233 <= line_num:
        should_indent = True
    
    if should_indent:
        # Don't indent empty lines if you prefer, but it doesn't hurt Python usually
        if content.strip() == "":
            new_lines.append(content)
        else:
            new_lines.append("    " + content)
    else:
        new_lines.append(content)

print("Writing fixed file...")
with open(file_path, 'w', encoding='utf-8') as f:
    f.writelines(new_lines)
print("Done.")
