from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

print(f"生成模型加载完成。{model_name}")

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

print(f"模型已移动到 {device}")

def generate_text(prompt, max_length=200, temperature=0.7, top_k=5):

    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    output = model.generate(
        input_ids,
        max_length=max_length,
        temperature=temperature,
        top_k=top_k,
        do_sample=True
    )

    answer = tokenizer.decode(output[0], skip_special_tokens=True)
    return answer


context = "糖尿病是一种慢性性疾病，需要长期控制血糖。"
query = "糖尿病患者如何管理饮食？"

prompt = f"{context} 问题: {query} 答案:"
answer = generate_text(prompt)

print(f"回答 {answer}")

generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0 if torch.cuda.is_available() else -1)
custom_answer = generator(prompt, max_length=200, num_return_sequences=1, temperature=0.6, top_k=5)[0]['generated_text']

print(f"自定义回答 {custom_answer}")


