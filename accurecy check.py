import torch

def calculate_accuracy(predictions, targets):
    correct = 0
    total = 0
    
    for pred, target in zip(predictions, targets):
        if pred == target:
            correct += 1
        total += 1
    
    accuracy = correct / total
    return accuracy

# Example test data (source, expected_output)
test_samples = [
    ([19, 16, 15, 45, 11, 8, 9, 32, 19, 7], [0, 19, 32, 9, 8, 11, 45, 15, 16, 19]),
    ([40, 5, 24, 26, 4, 34, 47, 7, 17, 30], [0, 17, 7, 47, 34, 4, 26, 24, 5, 40]),
    ([31, 21, 40, 27, 40, 43, 45, 21, 16, 48], [0, 16, 21, 45, 43, 40, 27, 40, 21, 31]),
    ([18, 6, 1, 14, 36, 23, 29, 4, 21, 10], [0, 21, 4, 29, 23, 36, 14, 1, 6, 18]),
    ([6, 24, 17, 27, 41, 15, 11, 12, 9, 28], [0, 9, 12, 11, 15, 41, 27, 17, 24, 6])
]

# Initialize lists to hold predictions and targets
predictions = []
targets = []

# Generate predictions and compare
for source, expected_output in test_samples:
    source_tensor = torch.tensor(source).unsqueeze(1).to(device)
    trg_tensor = torch.tensor(expected_output).unsqueeze(1).to(device)
    
    with torch.no_grad():
        output = model_with_attention(source_tensor, trg_tensor, teacher_forcing_ratio=0.0)
    
    # Flatten output and get predicted indices
    output = output.argmax(dim=-1).squeeze().tolist()
    
    # Collect predictions and expected outputs
    predictions.append(output)
    targets.append(expected_output)

# Calculate accuracy
accuracy = calculate_accuracy(predictions, targets)
print(f'Test Accuracy: {accuracy:.4f}')
