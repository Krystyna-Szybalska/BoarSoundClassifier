import os
import csv
import shutil

# Define the paths
csv_file_path = 'Metadata.csv'  # Update with the actual path to your CSV file
output_folder = 'Quality_0_Files'  # The folder to move the files with Quality = 0
source_directory = os.getcwd() + '\\PreparedData'

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Read the CSV file and process rows
with open(csv_file_path, mode='r', newline='', encoding='utf-8') as csv_file:
    csv_reader = csv.DictReader(csv_file, delimiter=';')

    for row in csv_reader:
        file_name = row['FileName'] + '.wav'
        quality = int(row['Quality'])

        if quality == 0:
            source_path = os.path.join(source_directory, file_name)
            destination_path = os.path.join(output_folder, file_name)

            # Move the file to the new folder
            try:
                shutil.move(source_path, destination_path)
                print(f"Moved '{file_name}' to '{output_folder}'")
            except FileNotFoundError:
                print(f"File '{file_name}' not found in the source directory.")

print("Done")