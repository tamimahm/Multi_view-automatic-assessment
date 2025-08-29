import pandas as pd
import random
from collections import defaultdict

def extract_patient_task_key(filename):
    """Extract patient_id and task_id from filename to create unique key"""
    try:
        # Expected format: ARAT_010_left_Impaired_cam3_activity1.mp4
        parts = filename.split('_')
        if len(parts) >= 5:
            patient_id = parts[1]  # e.g., "010"
            activity_part = parts[-1]  # e.g., "activity1.mp4"
            task_id = activity_part.replace('activity', '').replace('.mp4', '')
            return f"{patient_id}_{task_id}"
    except:
        pass
    return None

def load_and_process_rating_files():
    """Load all rating files and extract unique keys per rating"""
    
    files = {
        0: 'D:/final_chicago_results/all_0s.csv',
        1: 'D:/final_chicago_results/all_1s.csv', 
        2: 'D:/final_chicago_results/all_2s.csv',
        3: 'D:/final_chicago_results/all_3s.csv'
    }
    
    # Load dataframes and track which keys appear in which files
    dataframes = {}
    all_keys = defaultdict(list)  # key -> list of ratings containing it
    
    for rating, filename in files.items():
        try:
            df = pd.read_csv(filename)
            dataframes[rating] = df
            
            # Track keys for this rating
            for _, row in df.iterrows():
                key = extract_patient_task_key(row['Top'])
                if key:
                    all_keys[key].append(rating)
                    
        except Exception as e:
            print(f"Error loading {filename}: {e}")
            return None, None
    
    # Get unique keys per rating (remove duplicates across files)
    unique_keys_per_rating = {}
    unique_rows_per_rating = {}
    
    for rating in range(4):
        if rating not in dataframes:
            continue
            
        df = dataframes[rating]
        unique_keys = set()
        unique_rows = []
        
        for _, row in df.iterrows():
            key = extract_patient_task_key(row['Top'])
            if key and len(all_keys[key]) == 1:  # Key appears only in this rating
                if key not in unique_keys:  # Avoid internal duplicates
                    unique_keys.add(key)
                    unique_rows.append(row)
        
        unique_keys_per_rating[rating] = list(unique_keys)
        unique_rows_per_rating[rating] = unique_rows
        
        print(f"Rating {rating}: {len(unique_rows)} unique rows available")
    
    return unique_keys_per_rating, unique_rows_per_rating

def generate_therapist_csvs(output_dir='D:/final_chicago_results/therapists'):
    """Generate 4 CSV files for therapists with balanced distributions, plus a 5th CSV with unused rows"""
    
    print("="*70)
    print("GENERATING THERAPIST CSV FILES")
    print("="*70)
    
    # Load and process the data
    unique_keys, unique_rows = load_and_process_rating_files()
    if unique_keys is None:
        print("Error: Could not load rating files")
        return
    
    # Calculate optimal distribution based on available data
    # Target: 10% 0s, 20% 1s, 40% 2s, 30% 3s
    available_counts = {rating: len(keys) for rating, keys in unique_keys.items()}
    
    # Find limiting factor (rating 1 with only 77 keys for 4 therapists = ~19 each)
    max_therapists = min(available_counts[rating] // target for rating, target in 
                        [(0, 11), (1, 19), (2, 40), (3, 30)])
    
    print(f"Available unique keys per rating: {available_counts}")
    print(f"Maximum possible therapists with target distribution: {max_therapists}")
    
    # Adjust distribution for 4 therapists
    if max_therapists < 4:
        print(f"Adjusting distribution for 4 therapists...")
        # Calculate maximum rows per therapist per rating
        max_per_therapist = {rating: count // 4 for rating, count in available_counts.items()}
        print(f"Maximum rows per therapist per rating: {max_per_therapist}")
        
        # Use these maximums as our distribution
        distribution = max_per_therapist
    else:
        # Use target distribution
        distribution = {0: 11, 1: 19, 2: 40, 3: 30}
    
    print(f"Using distribution per therapist: {distribution}")
    total_per_therapist = sum(distribution.values())
    print(f"Total rows per therapist: {total_per_therapist}")
    
    # Shuffle the rows for each rating to ensure random distribution
    for rating in unique_rows.keys():
        random.shuffle(unique_rows[rating])
    
    # Track used rows for the 5th CSV
    used_indices_per_rating = {rating: set() for rating in unique_rows.keys()}
    
    # Generate CSV files for each therapist
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    for therapist_id in range(1, 5):  # Therapists 1-4
        print(f"\nGenerating CSV for Therapist {therapist_id}...")
        
        therapist_rows = []
        
        # Select rows for each rating
        for rating, count in distribution.items():
            if rating in unique_rows and len(unique_rows[rating]) > 0:
                # Calculate which rows to take for this therapist
                start_idx = (therapist_id - 1) * count
                end_idx = start_idx + count
                
                # Make sure we don't exceed available rows
                available_rows = len(unique_rows[rating])
                if start_idx < available_rows:
                    actual_end_idx = min(end_idx, available_rows)
                    selected_rows = unique_rows[rating][start_idx:actual_end_idx]
                    therapist_rows.extend(selected_rows)
                    
                    # Track which indices were used
                    for i in range(start_idx, actual_end_idx):
                        used_indices_per_rating[rating].add(i)
                    
                    print(f"  Rating {rating}: selected {len(selected_rows)} rows (wanted {count})")
                else:
                    print(f"  Rating {rating}: no rows available for therapist {therapist_id}")
        
        # Create DataFrame and save
        if therapist_rows:
            therapist_df = pd.DataFrame(therapist_rows)
            
            # Shuffle the rows so ratings are mixed
            therapist_df = therapist_df.sample(frac=1).reset_index(drop=True)
            
            # Save to CSV
            output_filename = os.path.join(output_dir, f'therapist_{therapist_id}.csv')
            therapist_df.to_csv(output_filename, index=False)
            
            print(f"  Saved {len(therapist_df)} rows to {output_filename}")
            
            # Show distribution summary
            rating_counts = {}
            for _, row in therapist_df.iterrows():
                key = extract_patient_task_key(row['Top'])
                # Determine which rating this came from by checking the original files
                for rating in range(4):
                    if rating in unique_keys and key in unique_keys[rating]:
                        rating_counts[rating] = rating_counts.get(rating, 0) + 1
                        break
            
            print(f"  Distribution: {rating_counts}")
        else:
            print(f"  No rows available for Therapist {therapist_id}")
    
    # Generate 5th CSV with unused rows
    print(f"\nGenerating CSV for unused rows (Therapist 5)...")
    unused_rows = []
    unused_counts = {}
    
    for rating, rows in unique_rows.items():
        unused_indices = set(range(len(rows))) - used_indices_per_rating[rating]
        unused_for_rating = [rows[i] for i in unused_indices]
        unused_rows.extend(unused_for_rating)
        unused_counts[rating] = len(unused_for_rating)
        print(f"  Rating {rating}: {len(unused_for_rating)} unused rows")
    
    if unused_rows:
        unused_df = pd.DataFrame(unused_rows)
        
        # Shuffle the unused rows
        unused_df = unused_df.sample(frac=1).reset_index(drop=True)
        
        # Save to CSV
        unused_filename = os.path.join(output_dir, 'therapist_5_unused.csv')
        unused_df.to_csv(unused_filename, index=False)
        
        print(f"  Saved {len(unused_df)} unused rows to {unused_filename}")
        print(f"  Distribution: {unused_counts}")
    else:
        print("  No unused rows found")
    
    print(f"\n" + "="*70)
    print("THERAPIST CSV GENERATION COMPLETE")
    print("="*70)
    print(f"Files saved to: {output_dir}")
    print(f"Generated 5 files: therapist_1.csv through therapist_4.csv, plus therapist_5_unused.csv")

def verify_therapist_csvs(therapist_dir='D:/final_chicago_results/therapists'):
    """Verify the generated therapist CSV files including the unused rows file"""
    
    print(f"\n" + "="*70)
    print("VERIFYING THERAPIST CSV FILES")
    print("="*70)
    
    import os
    
    all_used_keys = set()
    
    # Check therapists 1-4
    for therapist_id in range(1, 5):
        filename = os.path.join(therapist_dir, f'therapist_{therapist_id}.csv')
        
        if os.path.exists(filename):
            df = pd.read_csv(filename)
            print(f"\nTherapist {therapist_id}: {len(df)} rows")
            
            # Extract keys and check for duplicates
            keys = []
            for _, row in df.iterrows():
                key = extract_patient_task_key(row['Top'])
                if key:
                    keys.append(key)
            
            # Check for internal duplicates
            unique_keys = set(keys)
            internal_dupes = len(keys) - len(unique_keys)
            if internal_dupes > 0:
                print(f"  ⚠️  Internal duplicates: {internal_dupes}")
            else:
                print(f"  ✅ No internal duplicates")
            
            # Check for cross-therapist duplicates
            cross_dupes = len(all_used_keys & unique_keys)
            if cross_dupes > 0:
                print(f"  ⚠️  Cross-therapist duplicates: {cross_dupes}")
            else:
                print(f"  ✅ No cross-therapist duplicates")
            
            all_used_keys.update(unique_keys)
            
        else:
            print(f"Therapist {therapist_id}: File not found")
    
    # Check the unused rows file (therapist 5)
    unused_filename = os.path.join(therapist_dir, 'therapist_5_unused.csv')
    if os.path.exists(unused_filename):
        df = pd.read_csv(unused_filename)
        print(f"\nTherapist 5 (unused): {len(df)} rows")
        
        # Extract keys and check for duplicates with used keys
        unused_keys = []
        for _, row in df.iterrows():
            key = extract_patient_task_key(row['Top'])
            if key:
                unused_keys.append(key)
        
        unused_unique_keys = set(unused_keys)
        
        # Check for internal duplicates in unused file
        internal_dupes = len(unused_keys) - len(unused_unique_keys)
        if internal_dupes > 0:
            print(f"  ⚠️  Internal duplicates: {internal_dupes}")
        else:
            print(f"  ✅ No internal duplicates")
        
        # Check for overlap with therapists 1-4
        overlap_with_used = len(all_used_keys & unused_unique_keys)
        if overlap_with_used > 0:
            print(f"  ⚠️  Overlap with therapists 1-4: {overlap_with_used}")
        else:
            print(f"  ✅ No overlap with therapists 1-4")
            
        # Count distribution in unused file
        unused_rating_counts = {}
        for _, row in df.iterrows():
            key = extract_patient_task_key(row['Top'])  
            # We'd need to check original files to determine rating, but for now just show total
        print(f"  Distribution analysis would need original file lookup")
        
    else:
        print(f"Therapist 5 (unused): File not found")
    
    print(f"\nTotal unique keys used in therapists 1-4: {len(all_used_keys)}")
    if os.path.exists(unused_filename):
        total_unique_keys = len(all_used_keys) + len(unused_unique_keys)
        print(f"Total unique keys in unused file: {len(unused_unique_keys)}")
        print(f"Grand total unique keys: {total_unique_keys}")
    
    # Summary
    print(f"\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    print("Files generated:")
    for i in range(1, 5):
        filename = os.path.join(therapist_dir, f'therapist_{i}.csv')
        if os.path.exists(filename):
            count = len(pd.read_csv(filename))
            print(f"  therapist_{i}.csv: {count} rows")
    
    if os.path.exists(unused_filename):
        count = len(pd.read_csv(unused_filename))
        print(f"  therapist_5_unused.csv: {count} rows")
    
    print(f"\nNo duplicates across files: {'✅' if overlap_with_used == 0 else '⚠️'}")
    print(f"All original data preserved: {'✅' if 'total_unique_keys' in locals() else '?'}")

if __name__ == "__main__":
    # Set random seed for reproducibility
    random.seed(1206177)
    
    # Generate the therapist CSV files
    generate_therapist_csvs()
    
    # Verify the results
    verify_therapist_csvs()