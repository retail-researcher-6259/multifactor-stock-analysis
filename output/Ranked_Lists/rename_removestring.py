import os

# 1. Set the path to the folder containing your files
#    (Remember to use forward slashes / even on Windows)
folder_path = r'C:\Miscellaneous_Programs\Learnings\multifactor-program\output\Ranked_Lists\aaa'

# 2. Get a list of all files in the folder
try:
    files_in_folder = os.listdir(folder_path)
except FileNotFoundError:
    print(f"Error: The folder '{folder_path}' was not found. Please check the path.")
    exit()

# 3. Loop through each file and rename it if it contains "_manual"
print("Starting file renaming process...")
for filename in files_in_folder:
    if "_manual" in filename:
        # Create the old and new file names
        old_file_path = os.path.join(folder_path, filename)
        new_filename = filename.replace("_manual", "")
        new_file_path = os.path.join(folder_path, new_filename)

        # Rename the file
        os.rename(old_file_path, new_file_path)
        print(f"✅ Renamed: '{filename}'  ->  '{new_filename}'")

print("\nRenaming complete! 🎉")