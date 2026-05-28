import sys

import fire
import gradio as gr
import torch
import numpy as np
torch.set_num_threads(1)
import transformers
import json
import os

os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
from peft import PeftModel
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer
from sklearn.metrics import roc_auc_score

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except:  # noqa: E722
    pass


def recall_at_k(y_true, y_score, k):
    # 计算每个样本的前k个预测项
    top_k_indices = np.argsort(-y_score, axis=1)[:, :k]
    recalls = []

    for i in range(y_true.shape[0]):
        # 如果该样本没有正样本，跳过计算或赋值为0
        if np.sum(y_true[i]) == 0:
            recalls.append(0)
            continue

        # 捕获并处理错误
        try:
            true_labels = y_true[i, top_k_indices[i]]
        except Exception as e:
            print(f"Error: {e}")
            print(
                f"y_true shape: {y_true.shape}, y_score shape: {y_score.shape}, top_k_indices shape: {top_k_indices.shape}")
            return None

        # 计算每个样本的召回率，并加入到recalls列表中
        recall_value = np.sum(true_labels) / np.sum(y_true[i])
        recalls.append(recall_value)

    # 返回平均召回率
    return np.mean(recalls)

def mrr_at_k(y_true, y_score, k):
    mrrs = []
    for i in range(y_true.shape[0]):
        sorted_indices = np.argsort(-y_score[i])
        rank = 0
        for j in range(k):
            if y_true[i, sorted_indices[j]] == 1:
                rank = j + 1
                break
        mrrs.append(1.0 / rank if rank > 0 else 0)
    return np.mean(mrrs)


def ndcg_at_k(y_true, y_score, k):
    def dcg_at_k(r, k):
        """计算DCG@k的值"""
        r = np.asfarray(r)[:k]
        return np.sum(r / np.log2(np.arange(2, r.size + 2)))

    ndcgs = []
    for i in range(y_true.shape[0]):
        sorted_indices = np.argsort(-y_score[i])
        predicted_relevance = y_true[i, sorted_indices][:k]
        dcg_value = dcg_at_k(predicted_relevance, k)

        sorted_true_relevance = np.sort(y_true[i])[::-1]  # 从大到小排序
        idcg_value = dcg_at_k(sorted_true_relevance, k)

        ndcgs.append(dcg_value / (idcg_value + 1e-7))  # 防止除以0

    return np.mean(ndcgs)

def main(
        load_8bit: bool = False,
        base_model: str = "D:/mhf/TALLRec/decapoda-research/llama-7b-hf",
        lora_weights: str = "output/output8_book/LLM_ALLRec_75_5",
        test_data_path: str = "../data/book/test.json",
        result_json_data: str = "compare/compares.json",
        cutoff_len: int = 512,
        batch_size: int = 32,
        share_gradio: bool = False,
):
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='decapoda-research/llama-7b-hf'"

    model_type = lora_weights.split('/')[-1]
    parts = model_type.rsplit("_", 1)
    model_name = parts[0]
    NUM = parts[1]
    # NUM = 1
    if os.path.exists(result_json_data):
        f = open(result_json_data, 'r')
        data = json.load(f)
        f.close()
    else:
        data = dict()

    if model_name in data and NUM in data[model_name]:
        print("Key exists, exiting the program.")
        sys.exit(0)  # 退出程序

    if not data.__contains__(model_name):
        data[model_name] = {}
    if not data[model_name].__contains__(NUM):
        data[model_name][NUM] = {}

    tokenizer = LlamaTokenizer.from_pretrained(base_model)
    if device == "cuda":
        model = LlamaForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=load_8bit,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            torch_dtype=torch.float16,
            device_map={'': 0}
        )
    elif device == "mps":
        model = LlamaForCausalLM.from_pretrained(
            base_model,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
    else:
        model = LlamaForCausalLM.from_pretrained(
            base_model, device_map={"": device}, low_cpu_mem_usage=True
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            device_map={"": device},
        )

    tokenizer.padding_side = "left"
    # unwind broken decapoda-research config
    model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2

    if not load_8bit:
        model.half()  # seems to fix bugs for some users.

    model.eval()
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    def evaluate(
            instructions,
            inputs=None,
            temperature=0,
            top_p=1.0,
            top_k=40,
            num_beams=1,
            max_new_tokens=cutoff_len,
            batch_size=1,
            **kwargs,
    ):
        prompt = [generate_prompt(instruction, input) for instruction, input in zip(instructions, inputs)]
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(device)
        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams,
            **kwargs,
        )
        with torch.no_grad():
            generation_output = model.generate(
                **inputs,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=max_new_tokens,
                # batch_size=batch_size,
            )
        s = generation_output.sequences
        scores = generation_output.scores[0].softmax(dim=-1)
        # logits = torch.tensor(scores[:, [8241, 3782]], dtype=torch.float32).softmax(dim=-1)
        logits = scores[:, [8241, 3782]].clone().detach().softmax(dim=-1)

        yes_logits = logits[:, 0]
        no_logits = logits[:, 1]

        input_ids = inputs["input_ids"].to(device)
        L = input_ids.shape[1]
        s = generation_output.sequences
        output = tokenizer.batch_decode(s, skip_special_tokens=True)
        output = [_.split('Response:\n')[-1] for _ in output]

        return output, logits.tolist(), yes_logits.tolist()

    # testing code for readme
    logit_list = []
    gold_list = []
    outputs = []
    logits = []
    yes_logits = []
    from tqdm import tqdm
    gold = []
    pred = []
    yes_pred = []

    with open(test_data_path, 'r') as f:
        test_data = json.load(f)
        # test_data = test_data[:40]
        instructions = [_['instruction'] for _ in test_data]
        inputs = [_['input'] for _ in test_data]
        gold = [int(_['output'] == 'Yes.') for _ in test_data]

        def batch(list, batch_size=4):
            chunk_size = (len(list) - 1) // batch_size + 1
            for i in range(chunk_size):
                yield list[batch_size * i: batch_size * (i + 1)]

        for i, batch in tqdm(enumerate(zip(batch(instructions), batch(inputs)))):
            instructions, inputs = batch
            output, logit, yes_logit = evaluate(instructions, inputs)
            outputs = outputs + output
            logits = logits + logit
            yes_logits = yes_logits + yes_logit
        for i, test in tqdm(enumerate(test_data)):
            test_data[i]['predict'] = outputs[i]
            test_data[i]['logits'] = logits[i]
            pred.append(logits[i][0])
            yes_pred.append(yes_logits[i])

    from sklearn.metrics import roc_auc_score

    yes_probs = np.array(yes_pred)
    y_trues = np.array(gold)

    num = 20
    # 自动根据数据长度调整 reshape
    num_samples = len(y_trues) // num  # 确保 reshape 是合法的

    y_true = np.array(y_trues)[:num_samples * num].reshape(num_samples, num)  # 丢弃多余的部分
    y_score = np.array(yes_probs)[:num_samples * num].reshape(num_samples, num)

    k_list = [3, 5, 10]
    results = {}
    for k in k_list:
        results[f'recall@{k}'] = recall_at_k(y_true, y_score, k)
        results[f'mrr@{k}'] = mrr_at_k(y_true, y_score, k)
        results[f'ndcg@{k}'] = ndcg_at_k(y_true, y_score, k)
    results['auc'] = roc_auc_score(gold, pred)
    print(results)
    data[model_name][NUM] = results
    f = open(result_json_data, 'w')
    json.dump(data, f, indent=4)
    f.close()


def generate_prompt(instruction, input=None):
    if input:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.  # noqa: E501

### Instruction:
{instruction}

### Input:
{input}

### Response:
"""
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.  # noqa: E501

### Instruction:
{instruction}

### Response:
"""


if __name__ == "__main__":
    fire.Fire(main)