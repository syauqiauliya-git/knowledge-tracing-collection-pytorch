import pandas as pd
import os
from datetime import datetime

# Define file paths relative to the script's location
upload_folder = '/Users/syauqimuhammad/Documents/KULIAH/Skripsi/hcnohdKT/knowledge-tracing-collection-pytorch/datasets/datauploads'  # Folder for uploaded files
master_dataset_path = '/Users/syauqimuhammad/Documents/KULIAH/Skripsi/hcnohdKT/knowledge-tracing-collection-pytorch/datasets/MyClassroom/classroom_data.csv'  # Master dataset
biweekly_dataset_path = '/Users/syauqimuhammad/Documents/KULIAH/Skripsi/hcnohdKT/knowledge-tracing-collection-pytorch/datasets/databiweekly/biweeklydata.csv'  # Biweekly dataset file

def upload_new_data():
    # List all CSV files in the datauploads folder
    uploaded_files = [f for f in os.listdir(upload_folder) if f.endswith('.csv')]
    
    # If there are no files to process, exit the function
    if not uploaded_files:
        print("No CSV files found to upload.")
        return

    # Loop through all the CSV files in the upload folder
    for file_name in uploaded_files:
        file_path = os.path.join(upload_folder, file_name)
        
        # Load the uploaded batch (CSV file)
        new_batch = pd.read_csv(file_path)
        
        # Timestamp the data to track the batch
        new_batch['timestamp'] = datetime.now()
        
        # Load the existing biweekly dataset (if exists)
        if os.path.exists(biweekly_dataset_path):
            biweekly_data = pd.read_csv(biweekly_dataset_path)
        else:
            biweekly_data = pd.DataFrame()  # If no file exists, create an empty dataframe
        
        # Append new batch data to the biweekly dataset
        biweekly_data = pd.concat([biweekly_data, new_batch], ignore_index=True)
        
        # Save the updated biweekly dataset
        biweekly_data.to_csv(biweekly_dataset_path, index=False)
        print(f"New data from {file_name} uploaded and added to biweekly dataset.")
        
        # Delete the uploaded batch file after processing
        os.remove(file_path)
        print(f"Uploaded batch file deleted: {file_name}")

def create_biweekly_dataset():
    # Load the biweekly dataset
    biweekly_data = pd.read_csv(biweekly_dataset_path)
    
    # Group data by two-week periods
    biweekly_data['date'] = pd.to_datetime(biweekly_data['timestamp'])
    
    # Define the start date for the current period
    start_date = biweekly_data['date'].max() - pd.Timedelta(days=14)
    
    # Filter data for the past two weeks
    biweekly_data = biweekly_data[biweekly_data['date'] >= start_date]
    
    # Save the biweekly dataset
    biweekly_data.to_csv(biweekly_dataset_path, index=False)
    print(f"Biweekly dataset created: {biweekly_dataset_path}")

def verify_and_merge_biweekly_with_master():
    # Prompt the user for verification
    user_confirmation = input("Do you want to merge the biweekly dataset into the master dataset? (yes/no): ").strip().lower()

    if user_confirmation == 'yes':
        # Load the master dataset and biweekly dataset
        master_data = pd.read_csv(master_dataset_path)
        biweekly_data = pd.read_csv(biweekly_dataset_path)
        
        # Merge biweekly data with the master dataset (ensuring no duplicates)
        merged_data = pd.concat([master_data, biweekly_data], ignore_index=True).drop_duplicates()
        
        # Save the merged dataset into the master dataset file
        merged_data.to_csv(master_dataset_path, index=False)
        print(f"Biweekly data merged with master dataset: {master_dataset_path}")
        
        # Delete the biweekly dataset after merging
        os.remove(biweekly_dataset_path)
        print(f"Biweekly dataset deleted: {biweekly_dataset_path}")
    else:
        print("Merge operation cancelled.")

# Example usage:

# Step 1: Upload all new batches of data (This will be triggered automatically)
upload_new_data()  # This will process all CSV files in the datauploads folder

# Step 2: Create the biweekly dataset from the uploaded data
create_biweekly_dataset()

# Step 3: Verify and merge the biweekly dataset into the master dataset
verify_and_merge_biweekly_with_master()
