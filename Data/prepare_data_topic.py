import pandas as pd
import os
from pathlib import Path

def load_parquet_dir(directory):
    """Loads all parquet files in a directory into a single DataFrame."""
    directory = Path(directory)
    files = list(directory.glob("*.parquet"))
    
    if not files:
        print(f"Warning: No parquet files found in {directory}")
        return pd.DataFrame()
    
    print(f"Loading {len(files)} files from {directory}...")
    df_list = [pd.read_parquet(f) for f in files]
    return pd.concat(df_list, ignore_index=True)

def prepare_topic_data():
    # Set paths relative to this script's location
    current_dir = Path(__file__).parent
    base_path = current_dir / "Topic" / "MLSUM"
    output_path = current_dir / "Topic"
    
    # Ensure output directory exists
    output_path.mkdir(parents=True, exist_ok=True)
    
    splits = ["train", "validation", "test"]
    dfs = {}
    
    for split in splits:
        print(f"\nProcessing {split} split...")
        dir_path = base_path / split
        df = load_parquet_dir(dir_path)
        
        if not df.empty:
            # Solo lo guardamos en el diccionario para unificarlo después
            dfs[split] = df
            
            if 'topic' in df.columns:
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
        rename_map = {
            'text': 'Content',
            'title': 'Title',
            'topic': 'Topic'
        }
        
        # Check which columns exist before renaming
        existing_cols = {old: new for old, new in rename_map.items() if old in unified_df.columns}
        unified_df = unified_df.rename(columns=existing_cols)
        
        # Select and reorder columns
        columns_to_keep = ['Content', 'Title', 'Topic']
        # Only keep columns that actually exist
        columns_to_keep = [c for c in columns_to_keep if c in unified_df.columns]
        unified_df = unified_df[columns_to_keep]
        
        # Generate unique IDs in the format TOP_N
        unified_df.insert(0, 'ID', [f"TOP_{i+1}" for i in range(len(unified_df))])
        
        # Output: Save the unified dataset
        final_csv = output_path / "Dataset_Topic_Classifier.csv"
        unified_df.to_csv(final_csv, index=False)
        print(f"Saved unified dataset to {final_csv} with {len(unified_df)} rows.")
        
        if 'Topic' in unified_df.columns:
            print("\nFinal Topic distribution:")
            print(unified_df['Topic'].value_counts())
        
        print("\nFirst 5 rows of unified dataset:")
        print(unified_df.head())
    else:
        print("No data found to unify. Please check if the parquet files exist in " + str(base_path))

if __name__ == "__main__":
    prepare_topic_data()
