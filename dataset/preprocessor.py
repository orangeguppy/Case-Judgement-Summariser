import os

def convert_to_lowercase_and_clear_empty_lines(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        non_empty_lines = [line.lower() for line in lines if line.strip()]
    
    with open(file_path, 'w') as file:
        file.writelines(non_empty_lines)

def process_directory(directory_path):
    for filename in os.listdir(directory_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(directory_path, filename)
            convert_to_lowercase_and_clear_empty_lines(file_path)
            print(f"Processed {filename} - converted to lowercase and cleared empty lines.")


if __name__ == "__main__":
    # Change the directory accordingly
    target_directory = r'E:\AI\BIA\dataset\processed\IN-Abs\test-data\judgement'
    
    if os.path.exists(target_directory):
        process_directory(target_directory)
    else:
        print("Directory not found. Please provide a valid directory path.")
