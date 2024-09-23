import pickle
from tqdm import tqdm

data = pickle.load(open("experiment/modi_commonsensqa.pkl", 'rb'))
# Thresholds from 0.1 to 1.0 with an increment of 0.1
thresholds = [round(i * 0.1, 1) for i in range(1, 11)]

# Initialize a dictionary to store accuracies for each threshold
accuracy_results = {}

for threshold in tqdm(thresholds, desc="Processing thresholds"):
    correct_predictions = 0
    total_predictions = len(data)
    for key in data:
        item = data[key]
        confidence = item['confidence']
        pred_ans = item['pred_ans']
        revised_pred = item['revised_pred']
        believable = item['believable']
        gold_ans = item['gold_ans']
        
        # Decision logic based on threshold and believable flag
        if confidence >= threshold:
            final_pred = pred_ans
        else:
            if not believable and revised_pred is not None:
                final_pred = revised_pred
            else:
                final_pred = pred_ans
        
        # Compare the final prediction with the gold answer
        if final_pred == gold_ans:
            correct_predictions += 1
    
    # Calculate accuracy for the current threshold
    accuracy = correct_predictions / total_predictions
    accuracy_results[threshold] = accuracy

# Print the accuracy results
for threshold, accuracy in accuracy_results.items():
    print(f"Threshold: {threshold}, Accuracy: {accuracy}")