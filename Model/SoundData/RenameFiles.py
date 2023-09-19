import os

directory = "path_to_file"
string_to_remove = "string_to_remove"

for filename in os.listdir(directory):
    if string_to_remove in filename:
        new_filename = filename.replace(string_to_remove, "")
        current_path = os.path.join(directory, filename)
        new_path = os.path.join(directory, new_filename)
        os.rename(current_path, new_path)
