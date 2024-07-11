import os

def get_all_folder_paths(path):
    folder_paths = []
    for root, dirs, _ in os.walk(path):
        for dir_name in dirs:
            relative_path = os.path.relpath(os.path.join(root, dir_name), path)
            folder_paths.append(relative_path)
    return sorted(folder_paths)