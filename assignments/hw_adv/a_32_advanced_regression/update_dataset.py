import pandas as pd
import numpy as np
import os

# Locate the CSV file in the same directory as this script
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, 'apartamente_bucuresti.csv')

# Read the CSV
try:
    df = pd.read_csv(file_path)
    print(f"Successfully loaded {file_path}. Shape: {df.shape}")
except FileNotFoundError:
    print(f"Error: {file_path} not found.")
    exit(1)

# Generate synthetic data for 'mobilat'
# Options (lowercased and _ separated): 'mobilat', 'nemobilat'
mobilat_options = ['mobilat', 'nemobilat']
# Assume 70% are furnished
df['mobilat'] = np.random.choice(mobilat_options, size=len(df), p=[0.7, 0.3])

# Generate synthetic data for 'tip_incalzire'
# Options (lowercased and _ separated): 'centrala_proprie', 'termoficare', 'centrala_bloc', 'incalzire_pardoseala'
incalzire_options = ['centrala_proprie', 'termoficare', 'centrala_bloc', 'incalzire_pardoseala']
# Assumed probabilities
incalzire_probs = [0.6, 0.25, 0.1, 0.05]
df['tip_incalzire'] = np.random.choice(incalzire_options, size=len(df), p=incalzire_probs)

# Save the updated dataset
df.to_csv(file_path, index=False)
print(f"Updated dataset saved to {file_path}. New columns added: 'mobilat', 'tip_incalzire'.")
print("First 5 rows of new columns:")
print(df[['mobilat', 'tip_incalzire']].head())