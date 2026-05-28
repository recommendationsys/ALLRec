import os
import sys
from typing import List
import fire
import torch
import transformers
from datasets import load_dataset
import random
from peft import (  # noqa: E402
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
)
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from transformers import LlamaForCausalLM, LlamaTokenizer  # noqa: F402
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import Trainer_div

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class LearningNet(nn.Module):
    def __init__(self, hidden_size):
        super(LearningNet, self).__init__()
        self.GAP1 = nn.AdaptiveAvgPool1d(1)
        self.GAP2 = nn.AdaptiveAvgPool1d(1)
        self.GAP3 = nn.AdaptiveAvgPool1d(1)
        self.GAP4 = nn.AdaptiveAvgPool1d(1)

        hidden_size_reduction = hidden_size // 4

        self.FC1 = nn.Linear(hidden_size, hidden_size_reduction)
        self.FC2 = nn.Linear(hidden_size, hidden_size_reduction)
        self.FC3 = nn.Linear(hidden_size, hidden_size_reduction)
        self.FC4 = nn.Linear(hidden_size, hidden_size_reduction)
        self.FC5 = nn.Linear(hidden_size_reduction, hidden_size)

    def forward(self, x1, x2, x3, x4):
        x1 = self.GAP1(x1.permute(0, 2, 1)).squeeze(
            -1)  # (batch_size, hidden_size, seq_length)-> (batch_size, hidden_size, 1)
        x2 = self.GAP2(x2.permute(0, 2, 1)).squeeze(-1)
        x3 = self.GAP3(x3.permute(0, 2, 1)).squeeze(-1)
        x4 = self.GAP4(x4.permute(0, 2, 1)).squeeze(-1)

        x1 = torch.relu(self.FC1(x1))
        x2 = torch.relu(self.FC2(x2))
        x3 = torch.relu(self.FC3(x3))
        x4 = torch.relu(self.FC4(x4))

        # 计算平均值
        x_avg = (x1 + x2 + x3 + x4) / 4

        # 映射回 hidden_size
        module_features = self.FC5(x_avg)

        return module_features


def Kmeans_TSNE(name, arg_features, dimension):
    arg_features = arg_features.view(-1, dimension).cpu().numpy()

    # 设置聚类的数量，例如这里假设聚成 4 类
    n_clusters = 4
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(arg_features)

    # 进行 t-SNE 降维，降到 2D 进行可视化
    tsne = TSNE(n_components=2, random_state=42)
    tsne_results = tsne.fit_transform(arg_features)

    # 绘制 t-SNE 可视化图
    plt.figure(figsize=(8, 6))
    plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=cluster_labels, cmap='viridis', marker='o')
    plt.colorbar(label='Cluster Label')
    plt.title('t-SNE Visualization of Aggregated Features with KMeans Clustering')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.savefig(f"{name}.png", dpi=300)
    # plt.show()


# avg_diversity_loss
def diversity_loss(LLM_features, module_features, t=1.0):

    LLM_features = torch.stack(LLM_features)
    features = torch.cat((module_features, LLM_features), dim=0)

    L = module_features.shape[0]
    y_true = torch.arange(start=L, end=2*L)
    y_true = y_true.to(device)

    sim = torch.matmul(module_features, features.t())

    sim_self = torch.zeros_like(sim, dtype=torch.float32).cuda()

    for i in range(L):
        sim_self[i][i] = 1e12
    sim = sim - sim_self

    loss = F.cross_entropy(sim, y_true)

    return loss


def LossPredLoss(input, target, margin=0.5, reduction='mean'):
    assert len(input) % 2 == 0, 'the batch size is not even.'
    input = (input - input.flip(0))[:len(input) // 2]
    target = (target - target.flip(0))[:len(target) // 2]
    target = target.detach()
    one = 2 * torch.sign(torch.clamp(target, min=0)) - 1
    if reduction == 'mean':
        loss = torch.sum(torch.clamp(margin - one * input, min=0))
        loss = loss / input.size(0)
    elif reduction == 'none':
        loss = torch.clamp(margin - one * input, min=0)
    else:
        NotImplementedError()
    return loss


def train(
        # model/data params
        base_model: str = "D:/mhf/TALLRec/decapoda-research/llama-7b-hf",  # the only required argument
        train_data_path: str = "../data/movie/train.json",
        val_data_path: str = "../data/movie/valid.json",
        output_dir: str = "output/outputs",
        # training hyperparams
        sample: int = 64,
        seed: int = 2025,
        batch_size: int = 4,
        micro_batch_size: int = 4,
        num_epochs: int = 5,
        CYCLES: int = 6,
        UPDATE: int = 4,
        module_layer: str = None,
        lora_layer: str = "ALLRec-light",
        learning_rate: float = 3e-4,
        cutoff_len: int = 256,
        alpha: float = -1,
        llama_hidden_size: int = 4096,
        # lora hyperparams
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,

        # llm hyperparams
        train_on_inputs: bool = True,  # if False, masks out inputs in loss
        group_by_length: bool = False,  # faster, but produces an odd training loss curve
        # wandb params
        wandb_project: str = "",
        wandb_run_name: str = "",
        wandb_watch: str = "",  # options: false | gradients | all
        wandb_log_model: str = "",  # options: false | true
        resume_from_checkpoint: str = None,  # either training checkpoint or final adapter
):
    print(
        f"Training Alpaca-LoRA model with params:\n"
        f"base_model: {base_model}\n"
        f"train_data_path: {train_data_path}\n"
        f"val_data_path: {val_data_path}\n"
        f"sample: {sample}\n"
        f"seed: {seed}\n"
        f"output_dir: {output_dir}\n"
        f"batch_size: {batch_size}\n"
        f"micro_batch_size: {micro_batch_size}\n"
        f"num_epochs: {num_epochs}\n"

        f"CYCLES: {CYCLES}\n"
        f"UPDATE: {UPDATE}\n"
        f"module_layer: {module_layer}\n"
        f"lora_layer: {lora_layer}\n"

        f"learning_rate: {learning_rate}\n"
        f"cutoff_len: {cutoff_len}\n"
        f"alpha: {alpha}\n"
        f"llama_hidden_size: {llama_hidden_size}\n"
        f"lora_r: {lora_r}\n"
        f"lora_alpha: {lora_alpha}\n"
        f"lora_dropout: {lora_dropout}\n"

        f"train_on_inputs: {train_on_inputs}\n"
        f"group_by_length: {group_by_length}\n"
        f"wandb_project: {wandb_project}\n"
        f"wandb_run_name: {wandb_run_name}\n"
        f"wandb_watch: {wandb_watch}\n"
        f"wandb_log_model: {wandb_log_model}\n"
        f"resume_from_checkpoint: {resume_from_checkpoint}\n"
    )
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='decapoda-research/llama-7b-hf'"
    gradient_accumulation_steps = batch_size // micro_batch_size
    # print(f"gradient_accumulation_steps: {gradient_accumulation_steps}")

    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size

    # Check if parameter passed or if set within environ
    use_wandb = len(wandb_project) > 0 or (
            "WANDB_PROJECT" in os.environ and len(os.environ["WANDB_PROJECT"]) > 0
    )
    # Only overwrite environ if wandb param passed
    if len(wandb_project) > 0:
        os.environ["WANDB_PROJECT"] = wandb_project
    if len(wandb_watch) > 0:
        os.environ["WANDB_WATCH"] = wandb_watch
    if len(wandb_log_model) > 0:
        os.environ["WANDB_LOG_MODEL"] = wandb_log_model

    model = LlamaForCausalLM.from_pretrained(
        base_model,
        load_in_8bit=True,
        torch_dtype=torch.float16,
        device_map=device_map,
    )

    os.environ["WANDB_DISABLED"] = "true"

    tokenizer = LlamaTokenizer.from_pretrained(base_model)

    tokenizer.pad_token_id = (
        0  # unk. we want this to be different from the eos token
    )
    tokenizer.padding_side = "left"  # Allow batched inference

    def tokenize(prompt, add_eos_token=True):
        # there's probably a way to do this with the tokenizer settings
        # but again, gotta move fast
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=cutoff_len,
            padding=False,
            return_tensors=None,
        )
        if (
                result["input_ids"][-1] != tokenizer.eos_token_id
                and len(result["input_ids"]) < cutoff_len
                and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()

        return result

    def generate_and_tokenize_prompt(data_point):
        full_prompt = generate_prompt(data_point)
        tokenized_full_prompt = tokenize(full_prompt)
        if not train_on_inputs:
            user_prompt = generate_prompt({**data_point, "output": ""})
            tokenized_user_prompt = tokenize(user_prompt, add_eos_token=False)
            user_prompt_len = len(tokenized_user_prompt["input_ids"])
            # tokenized_full_prompt["last_position"] = user_prompt_len - 1
            tokenized_full_prompt["labels"] = [
                                                  -100
                                              ] * user_prompt_len + tokenized_full_prompt["labels"][
                                                                    user_prompt_len:
                                                                    ]  # could be sped up, probably
        return tokenized_full_prompt

    # 自定义的collate函数，将每个数据点转换为tensor并组合成batch
    def collate_fn(batch):
        input_ids = [torch.tensor(item['input_ids']).to(device) for item in batch]
        attention_masks = [torch.tensor(item['attention_mask']).to(device) for item in batch]
        labels = [torch.tensor(item['labels']).to(device) for item in batch]

        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=0)
        attention_masks = torch.nn.utils.rnn.pad_sequence(attention_masks, batch_first=True, padding_value=0)
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_masks,
            'labels': labels
        }

    def get_uncertainty(model, loss_module, unlabeled_data, get_module_features, alpha):
        # print(alpha)
        unlabeled_dataloder = DataLoader(unlabeled_data, batch_size=8, collate_fn=collate_fn)
        uncertainty = torch.tensor([]).cuda()
        LLM_features = torch.tensor([]).cuda()
        module_features = torch.tensor([]).cuda()
        cos = torch.tensor([]).cuda()
        loss = torch.tensor([]).cuda()
        for batch in tqdm(unlabeled_dataloder):
            inputs = batch["input_ids"]
            labels = batch["labels"].to(device)
            attention_mask = batch["attention_mask"]
            # with torch.no_grad():
            outputs = model(inputs, attention_mask, output_hidden_states=True)
            logits = outputs["logits"]
            logits = logits.detach()
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            # LLM's loss
            lm_loss = []
            for i in range(logits.size(0)):
                loss_ = loss_fct(shift_logits[i].float().view(-1, shift_logits.size(-1)), shift_labels[i].view(-1))
                lm_loss.append(loss_)
            # 将损失张量列表堆叠成一个张量
            pred_loss = torch.stack(lm_loss)

            # cos
            hidden_states = outputs.hidden_states
            features = [hidden_states[-4].detach(), hidden_states[-3].detach(), hidden_states[-2].detach(),
                        hidden_states[-1].detach()]
            features = [feature.float() for feature in features]
            module_feature = loss_module(*features)

            # cos
            features_norm = module_feature.norm(p=2, dim=1, keepdim=True)
            get_arg_features_norm = get_module_features.norm(p=2, dim=1, keepdim=True)
            cos_sim_matrix = torch.mm(module_feature, get_module_features.t())
            cos_sim_matrix = cos_sim_matrix / (features_norm * get_arg_features_norm.t())
            mean_cos_sim = cos_sim_matrix.mean(dim=1)

            score = pred_loss - alpha * mean_cos_sim
            # score = pred_loss - alpha2 * mean_cos_sim
            module_features = torch.cat((module_features, module_feature.detach()), 0)

            loss = torch.cat((loss, pred_loss), 0)
            cos = torch.cat((cos, mean_cos_sim), 0)

            uncertainty = torch.cat((uncertainty, score), 0)

            # 先进行平均池化
            pooled_features = torch.stack(features).permute(1, 0, 2, 3).mean(dim=2)
            LLM_features = torch.cat((LLM_features, pooled_features.detach().view(-1)), 0)
        # alpha2 = torch.sum(loss) / torch.sum(cos)
        return uncertainty, module_features, LLM_features

    model = prepare_model_for_int8_training(model)

    Learning_module = LearningNet(llama_hidden_size).to(device)

    if lora_layer == "ALLRec":
        lora_target_modules = [
            "q_proj",
            "v_proj",
        ]
    elif lora_layer == "ALLRec-light":
        # 定义要应用 LoRA 的层（1,3,5,...,31）
        lora_layers = list(range(16))

        # 创建目标模块列表，包含每层的 q_proj 和 v_proj 模块
        lora_target_modules = []
        for i in lora_layers:
            lora_target_modules.append(f"model.layers.{2 * i + 1}.self_attn.q_proj")
            lora_target_modules.append(f"model.layers.{2 * i + 1}.self_attn.k_proj")
            lora_target_modules.append(f"model.layers.{2 * i + 1}.self_attn.v_proj")

    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)

    if train_data_path.endswith(".json"):  # todo: support jsonl
        train_data = load_dataset("json", data_files=train_data_path)
    else:
        train_data = load_dataset(train_data_path)

    if val_data_path.endswith(".json"):  # todo: support jsonl
        val_data = load_dataset("json", data_files=val_data_path)
    else:
        val_data = load_dataset(val_data_path)

    train_data = (train_data["train"].map(generate_and_tokenize_prompt))
    val_data = (val_data["train"].map(generate_and_tokenize_prompt))

    if resume_from_checkpoint:
        # Check the available weights and load them
        checkpoint_name = os.path.join(
            resume_from_checkpoint, "pytorch_model.bin"
        )  # Full checkpoint
        if not os.path.exists(checkpoint_name):
            checkpoint_name = os.path.join(
                resume_from_checkpoint, "adapter_model.bin"
            )  # only LoRA model - LoRA config above has to fit
            resume_from_checkpoint = (
                False  # So the trainer won't try loading its state
            )
        # The two files above have a different name depending on how they were saved, but are actually the same.
        if os.path.exists(checkpoint_name):
            print(f"Restarting from {checkpoint_name}")
            adapters_weights = torch.load(checkpoint_name)
            model = set_peft_model_state_dict(model, adapters_weights)
        else:
            print(f"Checkpoint {checkpoint_name} not found")

    model.print_trainable_parameters()  # Be more transparent about the % of trainable params.

    if not ddp and torch.cuda.device_count() > 1:
        model.is_parallelizable = True
        model.model_parallel = True

    # 设置常量
    NUM_TRAIN = len(train_data)
    ADDENDUM = sample  # 选择你希望的数量
    # UPDATE = sample // 16
    UPDATE = UPDATE

    # 创建索引并打乱
    indices = list(range(NUM_TRAIN))
    random.shuffle(indices)

    # 分割为有标签和无标签的集合
    labeled_set = indices[:ADDENDUM]
    unlabeled_set = indices[ADDENDUM:]

    labeled_data = train_data.select(labeled_set)

    # Active learning cycles

    SUBSET = 1000
    # SUBSET = 16
    loss_fct = torch.nn.CrossEntropyLoss()

    for cycle in tqdm(range(CYCLES)):
        trainer = Trainer_div.CustomTrainer(
            model=model,
            Learning_module=Learning_module,
            diversity_loss=diversity_loss,
            train_dataset=labeled_data,
            eval_dataset=val_data,
            args=transformers.TrainingArguments(
                per_device_train_batch_size=micro_batch_size,
                gradient_accumulation_steps=gradient_accumulation_steps,
                warmup_steps=20,
                num_train_epochs=num_epochs,
                learning_rate=learning_rate,
                fp16=True,
                logging_steps=8,
                optim="adamw_torch",
                evaluation_strategy="steps",
                save_strategy="steps",
                eval_steps=200,
                save_steps=200,
                output_dir=output_dir,
                save_total_limit=1,
                load_best_model_at_end=True,
                ddp_find_unused_parameters=False if ddp else None,
                group_by_length=group_by_length,
                report_to=None,
                # report_to="wandb" if use_wandb else None,
                # run_name=wandb_run_name if use_wandb else None,
                # eval_accumulation_steps=10,
            ),
            data_collator=transformers.DataCollatorForSeq2Seq(
                tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
            ),
        )

        model.config.use_cache = False

        if torch.__version__ >= "2" and sys.platform != "win32":
            model = torch.compile(model)

        trainer.train(resume_from_checkpoint=resume_from_checkpoint)

        get_module_features = trainer.get_module_features()
        get_module_features = get_module_features.view(-1, 4096)[-sample:, :]

        # Upadate the train dataset
        if cycle != CYCLES - 1:
            # Randomly sample the unlabeled data points
            random.shuffle(unlabeled_set)
            subset = unlabeled_set[:SUBSET]

            unlabeled_data = train_data.select(subset)

            # Measure uncertainty of each data points in the subset
            uncertainty, _, _ = get_uncertainty(model, Learning_module, unlabeled_data, get_module_features, alpha)

            arg_loss = trainer.get_arg_loss().view(num_epochs, sample).mean(dim=0)

            # cos
            get_arg_features_norm = get_module_features.norm(p=2, dim=1, keepdim=True)
            cos_sim_matrix = torch.mm(get_module_features, get_module_features.t())
            cos_sim_matrix = cos_sim_matrix / (get_arg_features_norm * get_arg_features_norm.t())
            mean_cos_sim = cos_sim_matrix.mean(dim=1)

            score = arg_loss - alpha * mean_cos_sim

            arg1 = torch.argsort(score, descending=True).cpu()
            arg2 = torch.argsort(uncertainty).cpu()

            labeled_set = list(torch.tensor(labeled_set)[arg1][:-UPDATE].numpy()) + list(
                torch.tensor(subset)[arg2][-UPDATE:].numpy())
            unlabeled_set = list(torch.tensor(labeled_set)[arg1][-UPDATE:].numpy()) + list(
                torch.tensor(subset)[arg2][:-UPDATE].numpy()) + unlabeled_set[SUBSET:]

            labeled_data = train_data.select(labeled_set)

        if cycle == CYCLES - 1:
            # Randomly sample the unlabeled data points
            random.shuffle(unlabeled_set)
            subset = unlabeled_set[:SUBSET]

            unlabeled_data = train_data.select(subset)

            # Measure uncertainty of each data points in the subset
            uncertainty, module_features, LLM_features = get_uncertainty(model, Learning_module, unlabeled_data, get_module_features, alpha)

            # 保存 module_features 和 LLM_features
            torch.save(module_features, f'{output_dir}/module_features.pt')
            torch.save(LLM_features, f'{output_dir}/LLM_features.pt')

    # 保存模型
    model.save_pretrained(output_dir)
    loss_module_path = os.path.join(output_dir, 'Learning_module_state_dict.pth')
    torch.save(Learning_module.state_dict(), loss_module_path)

    print(
        "\n If there's a warning about missing keys above, please disregard :)"
    )


def generate_prompt(data_point):
    # sorry about the formatting disaster gotta move fast
    if data_point["input"]:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.  # noqa: E501

### Instruction:
{data_point["instruction"]}

### Input:
{data_point["input"]}

### Response:
{data_point["output"]}"""
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request. 

### Instruction:
{data_point["instruction"]}

### Response:
{data_point["output"]}"""


if __name__ == "__main__":
    fire.Fire(train)
