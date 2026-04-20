import pandas as pd
import os
import glob

def load_parquet_dir(directory):
    """Loads all parquet files in a directory into a single DataFrame."""
    files = glob.glob(os.path.join(directory, "*.parquet"))
    if not files:
        print(f"Warning: No parquet files found in {directory}")
        return pd.DataFrame()
    
    print(f"Loading {len(files)} files from {directory}...")
    df_list = [pd.read_parquet(f) for f in files]
    return pd.concat(df_list, ignore_index=True)

def prepare_topic_data():
    base_path = os.path.join("Data", "Topic", "MLSUM")
    output_path = os.path.join("Data", "Topic")
    
    # Ensure output directory exists
    os.makedirs(output_path, exist_ok=True)
    
    splits = ["train", "validation", "test"]
    dfs = {}
    
    for split in splits:
        print(f"\nProcessing {split} split...")
        dir_path = os.path.join(base_path, split)
        df = load_parquet_dir(dir_path)
        
        if not df.empty:
            # Save individual export
            csv_name = f"Dataset_Topic_{'val' if split == 'validation' else split}.csv"
            df.to_csv(os.path.join(output_path, csv_name), index=False)
            print(f"Saved {csv_name} with {len(df)} rows.")
            
            # Keep for unification
            dfs[split] = df
            
            print(f"Topic distribution for {split}:")
            print(df['topic'].value_counts())
        else:
            print(f"Skipping {split} because it's empty.")

    # Unification Logic
    if dfs:
        print("\nUnifying datasets...")
        unified_df = pd.concat(dfs.values(), ignore_index=True)
        
        # Normalization
        # Map text -> Content, title -> Title, topic -> Topic
        unified_df = unified_df.rename(columns={
            'text': 'Content',
            'title': 'Title',
            'topic': 'Topic'
        })
        
        # Select and reorder columns
        columns_to_keep = ['Content', 'Title', 'Topic']
        unified_df = unified_df[columns_to_keep]
        
        # Generate unique IDs in the format TOP_N
        unified_df.insert(0, 'ID', [f"TOP_{i+1}" for i in range(len(unified_df))])
        
        # Output: Save the unified dataset
        final_csv = os.path.join(output_path, "Dataset_Topic_Classifier.csv")
        unified_df.to_csv(final_csv, index=False)
        print(f"Saved unified dataset to {final_csv} with {len(unified_df)} rows.")
        
        print("\nFinal Topic distribution:")
        print(unified_df['Topic'].value_counts())
        
        print("\nFirst 5 rows of unified dataset:")
        print(unified_df.head())
    else:
        print("No data found to unify.")

if __name__ == "__main__":
    prepare_topic_data()
