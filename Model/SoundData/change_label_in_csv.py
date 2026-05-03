import os
import csv

txt_file = r"C:\Users\Krysia\Desktop\To pad and 0.txt"

audio_paths = []
with open(txt_file, 'r') as f:
    audio_paths = [line for line in f]

audio_paths = [os.path.basename(line) for line in audio_paths]
audio_paths = [os.path.splitext(line)[0] for line in audio_paths]
audio_file_stems_to_update = set(audio_paths)

csv_file_path = 'Metadata.csv'  # Update with the actual path to your CSV file



# 2. Read the CSV, modify relevant rows, and store them
modified_rows = []
field_names = None
rows_updated_count = 0

try:
    with open(csv_file_path, mode='r', newline='', encoding='utf-8') as infile:
        reader = csv.DictReader(infile, delimiter=';')
        field_names = reader.fieldnames

        if not field_names:
            print(f"Error: Could not read header from '{csv_file_path}'. Is the file empty or format incorrect?")
            exit(1)

        if 'FileName' not in field_names or 'Quality' not in field_names:
            print(f"Error: CSV file '{csv_file_path}' must contain 'FileName' and 'Quality' columns.")
            print(f"Found columns: {field_names}")
            exit(1)

        for row in reader:
            # The 'FileName' column in Metadata.csv already appears to be a stem
            current_file_name_stem = row.get('FileName')
            if current_file_name_stem in audio_file_stems_to_update:
                if row.get('Quality') != '0':  # Check if update is actually needed
                    row['Quality'] = '0'  # Update Quality to '0' (as a string)
                    rows_updated_count += 1
            modified_rows.append(row)

    if not modified_rows and field_names:  # Header was read, but no data rows
        print(f"Info: CSV file '{csv_file_path}' contains a header but no data rows.")


except FileNotFoundError:
    print(f"Error: The CSV file '{csv_file_path}' was not found.")
    exit(1)
except Exception as e:
    print(f"An error occurred while reading or processing '{csv_file_path}': {e}")
    exit(1)


# 3. Write the modified data (or original data if no changes were applicable) back to the CSV file
if field_names is not None:  # Proceed only if header was successfully read
    try:
        with open(csv_file_path, mode='w', newline='', encoding='utf-8') as outfile:
            writer = csv.DictWriter(outfile, fieldnames=field_names, delimiter=';')
            writer.writeheader()
            writer.writerows(modified_rows)

        if rows_updated_count > 0:
            print(
                f"Successfully updated '{csv_file_path}'. {rows_updated_count} row(s) had their 'Quality' changed to 0.")
        elif audio_file_stems_to_update:  # Targets existed, but no matches or no changes needed
            print(
                f"Processed '{csv_file_path}'. No rows required an update to 'Quality=0' based on the provided list, or they were already 0.")
        else:  # No targets to begin with
            print(
                f"Processed '{csv_file_path}'. The file was rewritten without changes as no target filenames were provided.")

    except Exception as e:
        print(f"An error occurred while writing the updated data to '{csv_file_path}': {e}")
else:
    # This path should ideally not be reached if exits above are working for header issues
    print("CSV processing aborted due to header not being available. File not written.")


