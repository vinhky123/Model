import kagglehub
import os
import shutil


def download_and_organize_pems():
    # Create main dataset directory
    os.makedirs("./dataset/PEMS", exist_ok=True)

    # Download the dataset
    print("Downloading PEMS dataset...")
    path = kagglehub.dataset_download("vinhky/pems-family")

    # List of PEMS folders to process
    pems_folders = ["PEMS03", "PEMS04", "PEMS07", "PEMS08"]

    # Process each PEMS folder
    for folder in pems_folders:
        # Create subfolder for each PEMS dataset
        target_folder = os.path.join("./dataset/PEMS", folder)
        os.makedirs(target_folder, exist_ok=True)

        source_folder = os.path.join(path, folder)
        if os.path.exists(source_folder):
            # Get all CSV files in the folder
            csv_files = [f for f in os.listdir(source_folder) if f.endswith(".csv")]

            # Move each CSV file to its corresponding subfolder
            for csv_file in csv_files:
                source_path = os.path.join(source_folder, csv_file)
                target_path = os.path.join(target_folder, csv_file)
                shutil.copy2(source_path, target_path)
                print(f"Copied {csv_file} to dataset/PEMS/{folder}/")

    print("Dataset download and organization completed!")


if __name__ == "__main__":
    download_and_organize_pems()
