import os
import glob
import re

def rename_files(folder_path, old_prefix, new_prefix):

    for filename in os.listdir(folder_path):
        file_path=folder_path+filename
        print(file_path)
        if filename.endswith(".txt"):

            match = re.match(rf'^{re.escape(old_prefix)}_(.*)$', filename)

            if match:
                remaining_part = match.group(1)
                new_file_name = f"{new_prefix}_{remaining_part}"
                new_file_path = os.path.join(folder_path, new_file_name)
                os.rename(file_path, new_file_path)



def replace_first_occurrence_in_file(input_file, output_file, x, y):
    with open(input_file, 'r') as file:
        lines = file.readlines()

    with open(output_file, 'w') as file:
        for line in lines:
            modified_line = line.replace(x, y, 1)
            file.write(modified_line)

if __name__ == "__main__":
    input_file = "input.txt"
    output_file = "output.txt"
    x = ""
    y = "new_string"

    replace_first_occurrence_in_file(input_file, output_file, x, y)

# if __name__ == "__main__":
#     folder_path = "predetections/" #Enter the path to the folder containing the text files: ")
#     old_prefix = "D4-S"#input("Enter the current prefix you want to replace: ")
#     new_prefix = "D4-S_mp4"#input("Enter the new prefix to replace the existing one: ")
#
#     rename_files(folder_path, old_prefix, new_prefix)
#     print("Files renamed successfully!")
