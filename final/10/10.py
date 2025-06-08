#使用chatgpt 看過且懂了

from transformers import pipeline

# 建立文字生成的 pipeline，使用預設的 GPT-2 模型
generator = pipeline('text-generation', model='gpt2')

# 輸入提示文字
prompt = "Hellow world ,"

# 生成文字
results = generator(prompt, max_new_tokens=50, truncation=True)

# 輸出生成結果
for i, result in enumerate(results):
    print(f"Generated text {i+1}:")
    print(result['generated_text'])
