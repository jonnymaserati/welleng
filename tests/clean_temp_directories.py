import os

# get the path of the current file
Path = os.path.dirname(os.path.abspath('__file__'))
# get the list of folders in the current path
folders = os.listdir(Path)
# get list of directories in the current path
directories = [folder for folder in folders if os.path.isdir(folder)]
# filter the directories that name starts within a certain string list
filtered_directories = [directory for directory in directories if directory.startswith(('cov_', 'error_'))]
print(filtered_directories)
print(len(filtered_directories))

# empty the filtered directories content
for directory in filtered_directories:
    for file in os.listdir(directory):
        os.remove(os.path.join(directory, file))
