import csv

# Define the paths
input_csv_path = 'Metadata.csv'  # Replace with the actual path to your CSV file
output_csv_path_0 = 'Metadata0.csv'
output_csv_path_1 = 'Metadata1.csv'

# Open the input CSV file for reading and the output CSV files for writing
with open(input_csv_path, mode='r', newline='', encoding='utf-8') as input_csv, \
        open(output_csv_path_0, mode='w', newline='', encoding='utf-8') as output_csv_0, \
        open(output_csv_path_1, mode='w', newline='', encoding='utf-8') as output_csv_1:
    # Create CSV readers and writers
    csv_reader = csv.DictReader(input_csv, delimiter=';')
    csv_writer_0 = csv.DictWriter(output_csv_0, fieldnames=csv_reader.fieldnames, delimiter=';')
    csv_writer_1 = csv.DictWriter(output_csv_1, fieldnames=csv_reader.fieldnames, delimiter=';')

    # Write headers to both output files
    csv_writer_0.writeheader()
    csv_writer_1.writeheader()

    # Iterate through the input CSV and split rows based on "Quality"
    for row in csv_reader:
        quality = int(row['Quality'])

        if quality == 0:
            csv_writer_0.writerow(row)
        else:
            csv_writer_1.writerow(row)

print("Splitting completed.")