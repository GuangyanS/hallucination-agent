import pickle
from transformers import AutoTokenizer
import numpy as np

# 读取pickle文件
with open('experiment/riddlesense.pkl', 'rb') as file:
    data = pickle.load(file)

def calculate_entropy(probabilities):
    """
    计算给定概率列表的熵
    """
    entropies = [p * np.exp(p) for p in probabilities]
    return -np.sum(entropies), -np.mean(entropies)
    
def average_entropy(data, correct_value):
    """
    计算给定correct_value (0或1)的平均熵
    """
    entropies, token_entropies = [], []
    
    for key, values in data.items():
        if values['correct'] == correct_value:
            prob = values['token_log_likelihoods']
            entropy, token_entropy = calculate_entropy(prob)
            entropies.append(entropy)
            token_entropies.append(token_entropy)
    
    if len(entropies) == 0:
        return 0
    return np.mean(entropies), np.mean(token_entropies)

# 计算0部分的平均熵
avg_entropy_0, avg_token_entropy_0 = average_entropy(data, 0)

# 计算1部分的平均熵
avg_entropy_1, avg_token_entropy_1 = average_entropy(data, 1)

# 打印结果
print(f"Average Entropy for incorrect (0): {avg_entropy_0:.3f}, Average Token Entropy for incorrect (0): {avg_token_entropy_0:.3f}")
print(f"Average Entropy for correct (1): {avg_entropy_1:.3f}, Average Token Entropy for correct (1): {avg_token_entropy_1:.3f}")
