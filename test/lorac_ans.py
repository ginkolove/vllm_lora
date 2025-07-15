from posixpath import split
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, DynamicCache, StaticCache
from datasets import load_dataset
import re
from peft import PeftModel
import torch.nn as nn

torch.cuda.empty_cache()
# 加载模型和 tokenizer
model_path = "meta-llama/Llama-2-7b-hf"
base_model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map="cuda:0")
tokenizer = AutoTokenizer.from_pretrained(model_path)
base_model.eval()
# 加载 ARC-Easy 数据集
dataset = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
sample = dataset[0]  # 示例样本

few_shot_examples = [
    {
        "question": "Which part of the plant conducts photosynthesis?",
        "choices": {
            "label": ["A", "B", "C", "D"],
            "text": ["Root", "Stem", "Leaf", "Flower"]
        },
        "answerKey": "C"
    },
    {
        "question": "Which of these objects is a gas at room temperature?",
        "choices": {
            "label": ["A", "B", "C", "D"],
            "text": ["Water", "Oxygen", "Ice", "Wood"]
        },
        "answerKey": "B"
    }
]
few_shot_prompt = ""
for ex in few_shot_examples:
    choices_formatted = "\n".join([f"{label}. {text}" for label, text in zip(ex["choices"]["label"], ex["choices"]["text"])])
    few_shot_prompt += f"Question: {ex['question']}\nChoices:\n{choices_formatted}\nAnswer: {ex['answerKey']}\n\n"

# 当前样本问题构造
question = sample["question"]
labels = sample["choices"]["label"]
texts = sample["choices"]["text"]
formatted_choices = "\n".join([f"{label}. {text}" for label, text in zip(labels, texts)])
answer_key = sample["answerKey"]

# 合成最终 prompt
final_prompt = f"{few_shot_prompt}Question: {question}\nChoices:\n{formatted_choices}\nAnswer"
base_cache = DynamicCache()
# 编码并生成
base_inputs = tokenizer(final_prompt, return_tensors="pt").to(base_model.device)
base_outputs = base_model.generate(**base_inputs, max_new_tokens=1, past_key_values=base_cache,do_sample=False)

base_decoded = tokenizer.decode(base_outputs[0], skip_special_tokens=True)

base_predicted_label = base_decoded[-1].strip().upper()
if base_predicted_label not in ["A", "B", "C", "D"]:
    base_predicted_label = "?"

print("Ground Truth Label:", answer_key)
print("\nBase Predicted Label:", base_predicted_label)
model = PeftModel.from_pretrained(base_model, "mogmyij/llama-arc-challenge-full-lora", device_map=base_model.device)
model.eval()
v_proj_inputs = []
v_proj_outputs = []
k_proj_inputs = []
k_proj_outputs = []
hooks = []
mooks=[]
for layer in model.base_model.model.model.layers:
    def get_hook():
        def hook_fn(module, input, output):
            v_proj_inputs.append(input[0].detach())
            v_proj_outputs.append(output.detach())
        return hook_fn

    h = layer.self_attn.v_proj.register_forward_hook(get_hook())
    hooks.append(h)
for layer in model.base_model.model.model.layers:
    def get_hook():
        def hook_fn(module, input, output):
            k_proj_inputs.append(input[0].detach())
            k_proj_outputs.append(output.detach())
        return hook_fn

    m = layer.self_attn.k_proj.register_forward_hook(get_hook())
    mooks.append(m)
lora_cache = DynamicCache()
lora_inputs = tokenizer(final_prompt, return_tensors="pt").to(base_model.device)
with torch.no_grad():
    lora_outputs = model.generate(**lora_inputs, max_new_tokens=1, past_key_values=lora_cache,do_sample=False)
lora_decoded = tokenizer.decode(lora_outputs[0], skip_special_tokens=True)
lora_predicted_label = lora_decoded[-1].strip().upper()
print("lora_predicted_label:", lora_predicted_label)
# model = PeftModel.from_pretrained(base_model, "domsoos/kgain-lora", device_map=base_model.device)
# model.eval()    
# lora_cache = DynamicCache()
# lora_inputs = tokenizer(final_prompt, return_tensors="pt").to(model.device)
mini_vs =[]
mini_ks =[]
recompute_vs=[]
recompute_ks=[]
sclaing = torch.tensor(2,dtype=torch.float32).to(base_model.device)
for i in range(len(v_proj_inputs)):
    mini_v = model.base_model.model.model.layers[i].self_attn.v_proj.lora_A['default'](v_proj_inputs[i].float())
    mini_k = model.base_model.model.model.layers[i].self_attn.k_proj.lora_A['default'](k_proj_inputs[i].float())
    mini_vs.append(mini_v)
    mini_ks.append(mini_k)
for i in range(len(v_proj_inputs)):
    # _, dequant_v = use(main_vs[i], base_cache.value_cache[i])
    # dequant_v =dequant_v.unsqueeze(0)
    # dequant_v = dequant_v.view(1,-1,4,128).permute(0,2,1,3).half()
    recompute_ks.append( base_cache.key_cache[i] +(sclaing*model.base_model.model.model.layers[i].self_attn.k_proj.lora_B['default'](mini_ks[i])).half().view(1,-1,32,128).permute(0,2,1,3) )
    recompute_vs.append( base_cache.value_cache[i] +(sclaing*model.base_model.model.model.layers[i].self_attn.v_proj.lora_B['default'](mini_vs[i])).half().view(1,-1,32,128).permute(0,2,1,3) )
lora_cache.value_cache = recompute_vs
lora_cache.key_cache = recompute_ks
with torch.no_grad():
    re_out = model.generate(
        input_ids=lora_outputs,
        past_key_values=lora_cache,
        max_new_tokens=1,
        do_sample=False
    )
#     logits=base_model(input_ids = lora_outputs[0][-1].unsqueeze(0).unsqueeze(0),past_key_values = lora_cache)["logits"]
#     out_token = torch.argmax(logits[-1,-1],dim=0,keepdim=True)

# reused_decoded = tokenizer.decode(out_token, skip_special_tokens=True)
# # print("peft输出:", )
reused_decoded = tokenizer.decode(re_out[0][-1], skip_special_tokens=True)
print("重用cache输出:", reused_decoded)
