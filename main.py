import imageio as iio


def binary_array_to_text(binary_array):
    # Flatten the array if it's multidimensional
    binary_array = binary_array.flatten()
    
    # Convert the binary array to a string of '0' and '1'
    binary_str = ''.join(binary_array.astype(str))
    
    # Split the binary string into groups of 8 bits
    n = 8
    byte_array = [binary_str[i:i + n] for i in range(0, len(binary_str), n)]
    
    # Convert each byte to an ASCII character
    text = ''.join([chr(int(byte, 2)) for byte in byte_array if len(byte) == 8])
    
    return text


def compare_files(file_path1, file_path2, output_path):
    # Open the two files and the output file
    with open(file_path1, 'r') as file1, open(file_path2, 'r') as file2, open(output_path, 'w') as output:
        # Read the contents of the files
        file1_lines = file1.readlines()
        file2_lines = file2.readlines()
        
        # Determine the maximum length to iterate through lines
        max_len = max(len(file1_lines), len(file2_lines))
        
        # Iterate through each line and compare
        for i in range(max_len):
            line1 = file1_lines[i].strip() if i < len(file1_lines) else ''
            line2 = file2_lines[i].strip() if i < len(file2_lines) else ''
            
            # If lines differ, write the difference to the output file
            if line1 != line2:
                output.write(f"Line {i+1}:\nFile 1: {line1}\nFile 2: {line2}\n\n")

 
 
# read an image 
img = iio.imread("stego.png")

binary_image = img[:, :, 3].flatten() // 255
print(binary_image[-100:])
hidden_text = binary_array_to_text(binary_array=binary_image)


f = open('out.txt', 'w')
f.write(hidden_text)
f.close()

# Compare the hidden text to the actual bee movie text
compare_files('out.txt', 'bee-movie.txt', 'diff.txt')
