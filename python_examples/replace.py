import os

def replace_string_in_file(file_path, old_string, new_string):
    """Replaces all occurrences of old_string with new_string in the file."""
    with open(file_path, 'r', encoding='utf-8') as file:
        file_data = file.read()

    # Replace the old string with the new one
    new_data = file_data.replace(old_string, new_string)

    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(new_data)

def replace_in_folder(folder_path, old_string, new_string):
    """Recursively goes through all files in a folder and replaces the string."""
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            
            # Only process text files (you can customize this to your needs)
            if file.endswith(('.txt', '.py', '.html', '.js', '.css', '.md')):
                try:
                    replace_string_in_file(file_path, old_string, new_string)
                    print(f"Replaced in: {file_path}")
                except Exception as e:
                    print(f"Failed to process {file_path}: {e}")

if __name__ == "__main__":
    folder_path = input("Enter the path to the folder: ")
    old_string = input("Enter the string to be replaced: ")
    new_string = input("Enter the new string: ")

    replace_in_folder(folder_path, old_string, new_string)
    print("Replacement complete.")
