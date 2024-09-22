import os

def replace_in_file(file_path, old_text, new_text):
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
        content = file.read()
    
    updated_content = content.replace(old_text, new_text)
    
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(updated_content)

def replace_in_folder(folder_path, old_text, new_text):
    for root, dirs, files in os.walk(folder_path):
        for file_name in files:
            if file_name.endswith('.html'):
                file_path = os.path.join(root, file_name)
                replace_in_file(file_path, old_text, new_text)

        for dir_name in dirs:
            subfolder_path = os.path.join(root, dir_name)
            replace_in_folder(subfolder_path, old_text, new_text)

folder_path = 'C:/Users/alexi/Documents/GitHub/ac-programming/'
old_text = 'ac-coding'
new_text = 'ac-programming'

replace_in_folder(folder_path, old_text, new_text)
