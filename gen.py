import numpy as np

# Function to generate synthetic dataset
def generate_synthetic_data(num_sequences, sequence_length, num_features):
    # Generate random sequences
    source_sequences = np.random.randint(0, num_features, (num_sequences, sequence_length))
    
    # Generate target sequences by reversing the source sequences
    target_sequences = np.flip(source_sequences, axis=1)
    
    return source_sequences, target_sequences

# Parameters
num_sequences = 10000  # Number of sequences
sequence_length = 10  # Length of each sequence
num_features = 50  # Number of unique integers in the sequences

# Generate the dataset
source_sequences, target_sequences = generate_synthetic_data(num_sequences, sequence_length, num_features)

# Split into training and testing sets
split_ratio = 0.8
split_index = int(num_sequences * split_ratio)

train_source = source_sequences[:split_index]
train_target = target_sequences[:split_index]
test_source = source_sequences[split_index:]
test_target = target_sequences[split_index:]

print("Training source sequences shape:", train_source.shape)
print("Training target sequences shape:", train_target.shape)
print("Testing source sequences shape:", test_source.shape)
print("Testing target sequences shape:", test_target.shape)

# Save the datasets
np.save('train_source.npy', train_source)
np.save('train_target.npy', train_target)
np.save('test_source.npy', test_source)
np.save('test_target.npy', test_target)
