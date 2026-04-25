import pandas as pd
import numpy as np
import json
import os

def prepare_sample_data():
    input_file = "data/input/test_split.csv"
    output_dir = "data_sample"
    output_file = os.path.join(output_dir, "test_sample.json")

    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading {input_file}...")
    df = pd.read_csv(input_file).fillna({"comment_text": "missing_text"})

    # Ensure necessary columns exist
    sub_cols = ['obscene', 'sexual_explicit', 'identity_attack', 'insult', 'threat']
    target_cols = ['toxicity'] + sub_cols
    
    for col in target_cols:
        if col not in df.columns:
            print(f"Error: Column '{col}' not found in dataset.")
            return

    print("Sampling non-toxic comments...")
    non_toxic = df[df['toxicity'] == 0]
    non_toxic_sample = non_toxic.sample(n=min(5000, len(non_toxic)), random_state=42)

    print("Sampling toxic comments (ensuring representation of subclasses)...")
    toxic = df[df['toxicity'] == 1]
    
    sampled_toxic_indices = set()
    toxic_samples = []

    # Try to get at least 500 of each subclass
    for sub_col in sub_cols:
        sub_df = toxic[(toxic[sub_col] == 1) & (~toxic.index.isin(sampled_toxic_indices))]
        sub_sample = sub_df.sample(n=min(500, len(sub_df)), random_state=42)
        sampled_toxic_indices.update(sub_sample.index)
        toxic_samples.append(sub_sample)

    # Combine sampled toxic subclasses
    toxic_sub_samples = pd.concat(toxic_samples) if toxic_samples else pd.DataFrame()
    
    # Fill the rest to reach 5000 toxic samples
    remaining_toxic_needed = max(0, 5000 - len(toxic_sub_samples))
    remaining_toxic = toxic[~toxic.index.isin(sampled_toxic_indices)]
    
    additional_toxic_sample = remaining_toxic.sample(n=min(remaining_toxic_needed, len(remaining_toxic)), random_state=42)
    
    final_toxic_sample = pd.concat([toxic_sub_samples, additional_toxic_sample])
    final_toxic_sample = final_toxic_sample.sample(n=min(5000, len(final_toxic_sample)), random_state=42) # Ensure we don't exceed 5000

    print(f"Collected {len(non_toxic_sample)} non-toxic and {len(final_toxic_sample)} toxic samples.")

    # Combine and shuffle
    final_sample = pd.concat([non_toxic_sample, final_toxic_sample]).sample(frac=1, random_state=42).reset_index(drop=True)

    # Convert to list of dicts
    records = final_sample[['comment_text'] + target_cols].to_dict(orient='records')

    print(f"Saving {len(records)} records to {output_file}...")
    with open(output_file, 'w') as f:
        json.dump(records, f, indent=2)
        
    print("Done!")

if __name__ == "__main__":
    prepare_sample_data()
