from transformers import Trainer, TrainingArguments, LlamaForCausalLM, LlamaTokenizer, get_linear_schedule_with_warmup
import torch
import torch.nn as nn
import copy
from transformers.modeling_utils import PreTrainedModel, load_sharded_checkpoint, unwrap_model
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES, MODEL_MAPPING_NAMES
device = "cuda" if torch.cuda.is_available() else "cpu"


class CustomTrainer(Trainer):
    def __init__(self,
                 *args,
                 Learning_module=None,
                 diversity_loss=None,
                 Learning_module_lr=1e-4,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.arg_loss = torch.tensor([]).cuda()
        self.arg_features = torch.tensor([]).cuda()
        self.arg_module_features = torch.tensor([]).cuda()
        self.diversity_loss = diversity_loss
        self.Learning_module = Learning_module
        self.Learning_module_lr = Learning_module_lr

        if self.Learning_module is not None and self.diversity_loss is not None:
            self.Learning_module.to(device)
            self.optimizer_Learning_module = torch.optim.AdamW(self.Learning_module.parameters(), lr=self.Learning_module_lr)
            self.lr_scheduler_Learning = get_linear_schedule_with_warmup(self.optimizer_Learning_module, num_warmup_steps=self.args.warmup_steps, num_training_steps=self.args.max_steps)
            self.Learning_module.train()

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """

        if not return_outputs:

            # 梯度置零
            if self.Learning_module is not None and self.diversity_loss is not None:
                self.optimizer_Learning_module.zero_grad()

            labels = inputs["labels"]
            outputs = model(**inputs, output_hidden_states=True)

            # Save past state if it exists
            # TODO: this needs to be fixed and made cleaner later.
            if self.args.past_index >= 0:
                self._past = outputs[self.args.past_index]

            logits = outputs["logits"]
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            loss_fct = torch.nn.CrossEntropyLoss()

            lm_loss = []
            for i in range(logits.size(0)):
                loss_ = loss_fct(shift_logits[i].float().view(-1, shift_logits.size(-1)), shift_labels[i].view(-1))
                lm_loss.append(loss_)
            # 将损失张量列表堆叠成一个张量
            lm_loss_tensor = torch.stack(lm_loss)
            # 计算平均损失
            average_loss = torch.mean(lm_loss_tensor)

            if self.arg_loss is not None:
                self.arg_loss = torch.cat((self.arg_loss, lm_loss_tensor.detach().view(-1)), 0)

            hidden_states = outputs.hidden_states
            features = [hidden_states[-4].detach(), hidden_states[-3].detach(), hidden_states[-2].detach(),
                        hidden_states[-1].detach()]

            pooled_features = torch.stack(features).permute(1, 0, 2, 3).mean(dim=2)
            self.arg_features = torch.cat((self.arg_features, pooled_features.detach().view(-1)), 0)

            if self.Learning_module is not None and self.diversity_loss is not None:
                module_features = self.Learning_module(*features)

                LLM_features = [torch.mean(feature, dim=0) for feature in hidden_states[-1].detach()]
                # 然后对池化后的特征进行聚合（拼接）

                self.arg_module_features = torch.cat((self.arg_module_features, module_features.detach().view(-1)), 0)

                module_loss = self.diversity_loss(LLM_features, module_features, t=1)

                module_loss.backward()

                # 在优化器步骤之前，我们使用梯度裁剪
                nn.utils.clip_grad_norm_(self.Learning_module.parameters(), max_norm=20, norm_type=2)

                self.optimizer_Learning_module.step()
                self.lr_scheduler_Learning.step()
        else:
            outputs = model(**inputs)
            average_loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        return (average_loss, outputs) if return_outputs else average_loss

    def get_arg_loss(self):
        return self.arg_loss

    def get_arg_features(self):
        return self.arg_features

    def get_module_features(self):
        return self.arg_module_features