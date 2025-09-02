import os
import re


def rename_stock_files(directory_path, insert_string="Steady_Growth"):
    """
    Rename files from 'top_ranked_stocks_MMDD.csv' to 'top_ranked_stocks_{insert_string}_MMDD.csv'

    Args:
        directory_path (str): Path to the directory containing the files
        insert_string (str): String to insert between 'top_ranked_stocks_' and the date
    """

    # Pattern to match files like 'top_ranked_stocks_0611.csv'
    pattern = r'^top_ranked_stocks_(\d{4})\.csv$'

    # Get all files in the directory
    files = os.listdir(directory_path)

    renamed_count = 0

    for filename in files:
        match = re.match(pattern, filename)
        if match:
            date_part = match.group(1)  # Extract the date part (e.g., '0611')

            # Create new filename
            new_filename = f"top_ranked_stocks_{insert_string}_{date_part}.csv"

            # Full paths
            old_path = os.path.join(directory_path, filename)
            new_path = os.path.join(directory_path, new_filename)

            # Check if new filename already exists
            if os.path.exists(new_path):
                print(f"Warning: {new_filename} already exists. Skipping {filename}")
                continue

            # Rename the file
            try:
                os.rename(old_path, new_path)
                print(f"Renamed: {filename} → {new_filename}")
                renamed_count += 1
            except Exception as e:
                print(f"Error renaming {filename}: {e}")

    print(f"\nCompleted! Renamed {renamed_count} files.")


# Usage example
if __name__ == "__main__":
    # Specify the directory containing your CSV files
    folder_path = "./Steady_Growth"  # Current directory - change this to your folder path

    # You can also change the insert string here if needed
    insert_text = "Steady_Growth"

    print(f"Renaming files in: {os.path.abspath(folder_path)}")
    print(f"Insert string: {insert_text}")
    print("-" * 50)

    rename_stock_files(folder_path, insert_text)