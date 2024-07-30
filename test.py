import pickle
from transformers import AutoTokenizer
import numpy as np
from tqdm import tqdm

tokenizer = AutoTokenizer.from_pretrained("/wudi/gysun/init_weights/Qwen2-0.5B")
data = pickle.load(open("experiment/riddlesense.pkl", 'rb'))


from lmdeploy import pipeline, GenerationConfig, TurbomindEngineConfig, ChatTemplateConfig

backend_config = TurbomindEngineConfig(model_name="internlm2-chat-7b", tp=1)
pipe = pipeline("/wudi/gysun/init_weights/anah-v2",
                 chat_template_config=ChatTemplateConfig(model_name='internlm2-chat-7b'), 
                 backend_config=backend_config)

template = """
You will act as a ’Hallucination’ annotator. I will provide you with a question, a partial answer to that question,
and related reference points. You need to determine whether the provided answer contains any hallucinatory
content and annotate the type of hallucination.
’Hallucination’ refers to content that contradicts the reference points or is unsupported by them.
## Judgment Criteria:
1. No Hallucination: If the answer is completely consistent with the reference points and does not introduce any
contradictory information, output: <No Hallucination>.
2. Contradiction: If the answer clearly contradicts the reference points, output: <Contradictory>.
3. Unverifiable: If the answer contains information not mentioned in the reference points and cannot be supported
or verified by them, output: <Unverifiable>.
## Task Process:
1. Carefully read the question, which is as follows: {question}
2. Carefully read the partial answer, which is as follows: {annotation}
3. Carefully read the reference points, which are as follows: Gold Answer: {reference}
4. Conduct the analysis: Based on the above judgment criteria, determine if the answer contains hallucinations
and output the type of hallucination.
"""
for index in tqdm(range(len(data))):
    filled_template = template.format(question=data[index]['question'], annotation=data[index]['rationale'], reference=data[index]['gold_ans'])
    print(80*'*')
    response = pipe([filled_template])
    print(data[index]['question'])
    print(data[index]['gold_ans'], data[index]['pred_ans'])
    print(data[index]['rationale'])
    print(response)