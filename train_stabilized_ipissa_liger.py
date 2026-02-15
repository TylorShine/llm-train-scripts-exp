# /// script
# requires-python = ">=3.9"
# dependencies = [
#   # --- Core ML/DL Frameworks ---
#   "torch>=2.1",
#   # --- Hugging Face Ecosystem ---
#   "transformers>=4.38",
#   "datasets>=2.18",
#   "peft>=0.9",
#   "accelerate>=0.27",
#   # --- Custom/High-Performance Components ---
#   "schedulefree",
#   # "flash-attn>=2.5.0",  # need to be install manually with --no-build-isolation
#   "liger-kernel",
#   # --- Utilities & Logging ---
#   "tensorboard",
#   "wandb",
#   "tqdm",
#   "pandas",
# ]
# ///

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, TaskType, get_peft_model
from peft.tuners.lora import Linear as LoraLinear
from liger_kernel.transformers import AutoLigerKernelForCausalLM
from datasets import load_dataset, concatenate_datasets
import schedulefree
import argparse
import os
import types
import gc

# --- 1. Inverse PiSSA (Tail-Tuning) 実装 ---

class InversePiSSALayer(nn.Module):
    """
    Inverse PiSSA: 主要成分(Top Singular Values)を固定し、
    微小成分(Tail Singular Values)のみを学習する層。
    """
    def __init__(self, original_linear, rank=128, alpha=1.0):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        
        # 元の重み情報を取得
        # LigerKernel等の特殊なLinear層の場合も .weight があれば動作する想定
        W = original_linear.weight.data.float() # SVD精度確保のためfloat32
        device = W.device
        dtype = original_linear.weight.dtype # 元の精度 (bf16/fp16)

        print(f"  > SVD computing for shape {W.shape} on {device}...")
        
        # 1. SVD実行 (メモリ不足時はCPUフォールバック)
        try:
            U, S, Vh = torch.linalg.svd(W, full_matrices=False)
        except torch.OutOfMemoryError:
            print("  > Warning: GPU OOM for SVD, falling back to CPU.")
            W_cpu = W.cpu()
            U, S, Vh = torch.linalg.svd(W_cpu, full_matrices=False)
            U, S, Vh = U.to(device), S.to(device), Vh.to(device)

        # 2. 成分分離
        # Inverse PiSSA: 下位 rank 個を学習対象(LoRA)にする
        # Keep indices: 0 ~ (Total - rank)
        keep_indices = slice(0, -rank)
        train_indices = slice(-rank, None)

        # Base (Frozen High-Energy components)
        W_base = U[:, keep_indices] @ torch.diag(S[keep_indices]) @ Vh[keep_indices, :]
        
        # Adapter Init (Trainable Low-Energy components)
        U_tail = U[:, train_indices]
        S_tail = S[train_indices]
        Vh_tail = Vh[train_indices, :]
        
        S_sqrt = torch.diag(torch.sqrt(S_tail))
        A_init = U_tail @ S_sqrt
        B_init = S_sqrt @ Vh_tail

        # 3. パラメータ登録
        # Baseは勾配計算しないバッファとして登録
        self.register_buffer('weight_base', W_base.to(dtype=dtype))
        
        # 学習対象のアダプタ (A: out x r, B: r x in)
        self.lora_A = nn.Parameter(A_init.to(dtype=dtype))
        self.lora_B = nn.Parameter(B_init.to(dtype=dtype))
        
        if original_linear.bias is not None:
            self.bias = nn.Parameter(original_linear.bias.data)
        else:
            self.register_parameter('bias', None)

        # メモリ掃除
        del W, U, S, Vh, W_base, A_init, B_init
        torch.cuda.empty_cache()

    def forward(self, x):
        # Base (Frozen) path
        base_out = nn.functional.linear(x, self.weight_base, self.bias)
        # Adapter (Trainable Tail) path: x @ B.T @ A.T
        adapter_out = (x @ self.lora_B.T) @ self.lora_A.T
        
        return base_out + (self.alpha * adapter_out)

    def merge_to_linear(self):
        """標準的なnn.Linearに戻す"""
        with torch.no_grad():
            W_new = self.weight_base + (self.lora_A @ self.lora_B)
            
            new_linear = nn.Linear(
                in_features=W_new.shape[1],
                out_features=W_new.shape[0],
                bias=(self.bias is not None)
            )
            new_linear.weight.data = W_new
            if self.bias is not None:
                new_linear.bias.data = self.bias
            return new_linear

def apply_inverse_pissa(model, target_modules=["o_proj", "down_proj"], rank=128):
    """モデル内の指定層をInversePiSSALayerに置換"""
    print(f"Converting target modules {target_modules} to Inverse PiSSA (Rank {rank})...")
    
    # 再帰的にモジュールを探索して置換
    # named_modules()だと置換中にイテレータが壊れる可能性があるため、名前リストを先に作る
    modules_to_replace = []
    for name, module in model.named_modules():
        if any(t in name for t in target_modules) and isinstance(module, (nn.Linear, LoraLinear)): 
            # LigerのLinearやLoRA層も対象にする場合、ここを調整
            # LigerKernelForCausalLMの場合、Linearは通常のnn.Linearではない可能性があるが、
            # .weight属性があれば動作するようにInversePiSSALayerを作ってある
            if hasattr(module, "weight"):
                modules_to_replace.append(name)
    
    for name in modules_to_replace:
        # 親モジュールと属性名を取得
        if "." in name:
            parent_name, child_name = name.rsplit(".", 1)
            parent = model.get_submodule(parent_name)
        else:
            parent = model
            child_name = name
            
        module = getattr(parent, child_name)
        print(f" - Processing: {name}")
        
        # 新しい層を作成 (ここでSVDが走る)
        pissa_layer = InversePiSSALayer(module, rank=rank)
        
        # 置換
        setattr(parent, child_name, pissa_layer)
        
    return model


# --- 2. 安定化ラッパー (位置認識型BPF損失の実装) ---

class StabilizedLigerWrapper(nn.Module):
    def __init__(self, model, init_scale=0.1, 
                 lambda_lpf=0.0, lambda_hpf=0.0, lambda_anchor=0.0, lambda_unitary=1e-3):
        super().__init__()
        self.model = model
        self.config = model.config
        
        self.lambda_lpf = lambda_lpf
        self.lambda_hpf = lambda_hpf
        self.lambda_anchor = lambda_anchor
        self.lambda_unitary = lambda_unitary
        
        self.tau_lpf = 0.8
        self.tau_hpf = 0.95
        self.anchor_temp = 0.1

        self.scales = nn.ParameterDict()
        
        # モデル構造の特定
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            self.layers = model.model.layers
        elif hasattr(model, "base_model") and hasattr(model.base_model.model, "model"): # LoRA Case
            self.layers = model.base_model.model.model.layers
        else: # Generic Fallback
            self.layers = model.model.layers
            
        self.num_layers = len(self.layers)
        
        for i in range(self.num_layers):
            self.scales[f"layer_{i}_attn"] = nn.Parameter(torch.ones(1) * init_scale)
            self.scales[f"layer_{i}_mlp"] = nn.Parameter(torch.ones(1) * init_scale)

        self._apply_forward_patch()
    
    def _apply_forward_patch(self):
        """実行時にパラメータを動的取得するパッチ"""
        for i, layer in enumerate(self.layers):
            # Attention Output (o_proj)
            if hasattr(layer.self_attn, "o_proj"):
                original_o_proj_forward = layer.self_attn.o_proj.forward
                key_attn = f"layer_{i}_attn"

                def new_o_proj_forward(module_self, x, *args, wrapper_ref=self, param_key=key_attn, orig_f=original_o_proj_forward, **kwargs):
                    out = orig_f(x, *args, **kwargs)
                    scale = wrapper_ref.scales[param_key]
                    return out * scale

                layer.self_attn.o_proj.forward = types.MethodType(new_o_proj_forward, layer.self_attn.o_proj)

            # MLP Output (down_proj)
            if hasattr(layer.mlp, "down_proj"):
                original_down_proj_forward = layer.mlp.down_proj.forward
                key_mlp = f"layer_{i}_mlp"

                def new_down_proj_forward(module_self, x, *args, wrapper_ref=self, param_key=key_mlp, orig_f=original_down_proj_forward, **kwargs):
                    out = orig_f(x, *args, **kwargs)
                    scale = wrapper_ref.scales[param_key]
                    return out * scale

                layer.mlp.down_proj.forward = types.MethodType(new_down_proj_forward, layer.mlp.down_proj)
                
    def get_input_embeddings(self): return self.model.get_input_embeddings()
    def set_input_embeddings(self, value): self.model.set_input_embeddings(value)
    def get_output_embeddings(self): return self.model.get_output_embeddings()
    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)

    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        need_hidden_states = (self.lambda_lpf > 0 or self.lambda_hpf > 0 or self.lambda_anchor > 0)
        
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=need_hidden_states, 
            return_dict=True,
            **kwargs
        )
        
        loss = None
        if labels is not None:
            # Base LM Loss
            logits = outputs.logits
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss(reduction='none')
            raw_loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))
            raw_loss = raw_loss.view(shift_labels.shape)
            
            # Position-aware weighting
            seq_len = raw_loss.shape[1]
            t = torch.linspace(0, 1, steps=seq_len, device=raw_loss.device)
            alpha = 1.07
            raw_weights = torch.exp(alpha * t)
            pos_weights = raw_weights / raw_weights.mean()
            pos_weights = pos_weights.unsqueeze(0)
            
            if attention_mask is not None:
                active_mask = attention_mask[..., 1:].float()
                weighted_loss = (raw_loss * pos_weights * active_mask).sum() / active_mask.sum().clamp(min=1)
            else:
                weighted_loss = (raw_loss * pos_weights).mean()
            loss = weighted_loss

            # Stability Losses (LPF, HPF, Anchor)
            if need_hidden_states:
                last_hidden_state = outputs.hidden_states[-1]
                batch_size, seq_full_len, _ = last_hidden_state.shape
                pos_ratio = torch.linspace(0, 1, steps=seq_full_len, device=last_hidden_state.device).unsqueeze(0)
                mask_float = attention_mask.float() if attention_mask is not None else torch.ones((batch_size, seq_full_len), device=last_hidden_state.device)

                # LPF
                if self.lambda_lpf > 0:
                    w_lpf = (1.0 - pos_ratio).clamp(min=0.0)
                    diff = last_hidden_state[:, 1:, :] - last_hidden_state[:, :-1, :]
                    diff_sq = diff.sum(dim=-1)
                    mask_lpf = mask_float[:, 1:] 
                    loss_lpf_raw = (diff_sq * w_lpf[:, 1:] * mask_lpf).sum() / mask_lpf.sum().clamp(min=1)
                    loss += self.lambda_lpf * F.relu(loss_lpf_raw**2 - self.tau_lpf)

                # HPF
                if self.lambda_hpf > 0:
                    w_hpf = pos_ratio
                    h_norm = F.normalize(last_hidden_state, p=2, dim=-1)
                    window = 5
                    if seq_full_len > window:
                        curr_h = h_norm[:, window:, :]
                        local_sims = torch.zeros((batch_size, seq_full_len - window), device=h_norm.device)
                        for w in range(1, window + 1):
                            past_h = h_norm[:, window-w : seq_full_len-w, :]
                            local_sims += (curr_h * past_h).sum(dim=-1)
                        local_sims /= window
                        mask_hpf = mask_float[:, window:]
                        loss_hpf_raw = (local_sims * w_hpf[:, window:] * mask_hpf).sum() / mask_hpf.sum().clamp(min=1)
                        loss += self.lambda_hpf * F.relu(loss_hpf_raw - self.tau_hpf)

                # Anchor
                if self.lambda_anchor > 0:
                    w_anchor = (pos_ratio ** 2)
                    anchor_window = min(5, seq_full_len)
                    h_anchor = last_hidden_state[:, :anchor_window, :].mean(dim=1, keepdim=True)
                    h_anchor_norm = F.normalize(h_anchor, p=2, dim=-1)
                    h_curr_norm = F.normalize(last_hidden_state, p=2, dim=-1)
                    pos_sim = torch.matmul(h_curr_norm, h_anchor_norm.transpose(1, 2)).squeeze(-1) / self.anchor_temp
                    
                    if batch_size > 1:
                        h_anchor_flat = h_anchor_norm.squeeze(1) 
                        all_sim = torch.matmul(h_curr_norm, h_anchor_flat.T) / self.anchor_temp
                        log_sum_exp_all = torch.logsumexp(all_sim, dim=-1)
                        loss_anchor_raw = - (pos_sim - log_sum_exp_all)
                    else:
                        loss_anchor_raw = 1.0 - torch.tanh(pos_sim * self.anchor_temp)
                    
                    loss += self.lambda_anchor * (loss_anchor_raw * w_anchor * mask_float).sum() / mask_float.sum().clamp(min=1)

        return (loss, outputs.logits) if loss is not None else outputs.logits
    
    def state_dict(self, destination=None, prefix='', keep_vars=False):
        """
        Trainerがチェックポイント保存時に呼び出すメソッド。
        全パラメータではなく、'requires_grad=True' (学習対象) のパラメータのみを返すことで、
        1. 共有メモリ(Tied Weights)エラーを回避
        2. チェックポイントサイズを劇的に削減 (GB単位 -> MB単位)
        """
        if destination is None:
            destination = {}

        # 通常の super().state_dict() は呼ばず、学習対象のみを手動で収集する
        for name, param in self.named_parameters():
            if param.requires_grad:
                # パラメータ名をキーとして保存 (prefix対応)
                destination[prefix + name] = param if keep_vars else param.detach()
        
        return destination

    def load_state_dict(self, state_dict, strict=True):
        """
        学習対象のみ保存された軽量チェックポイントをロードするための処理。
        保存データにはfrozenパラメータが含まれていないため、必ず strict=False として扱う必要があります。
        """
        # 親クラスの load_state_dict を strict=False で呼び出すことで、
        # state_dict に含まれていないキー（Frozenパラメータ）があってもエラーにしない
        
        # ログへの警告を抑制したい場合は、ここで keys を比較して missing_keys を握りつぶすこともできますが、
        # 基本的には PyTorch の標準挙動に任せつつ strict=False にするのが安全です。
        
        return super().load_state_dict(state_dict, strict=False)
    
    def save_merged_model(self, output_dir, tokenizer):
        """
        全パラメータのマージと保存を行う。
        処理順序:
          1. (もしあれば) LoRA PEFTアダプタのマージ
          2. (もしあれば) Inverse PiSSA層のマージ (Custom Layer -> nn.Linear)
          3. LayerScale (Residual Damping) の適用
        """
        print(f"Starting merge process for {output_dir}...")
        
        # 1. PEFT (LoRA) マージ
        if getattr(self.model, "peft_config", None) is not None:
            print(" -> Merging PEFT/LoRA adapters...")
            model_merged = self.model.merge_and_unload()
        else:
            model_merged = self.model

        # 2. Inverse PiSSA マージ
        # モデル内の InversePiSSALayer を探して置換する
        pissa_merged_count = 0
        modules_to_restore = []
        for name, module in model_merged.named_modules():
            if isinstance(module, InversePiSSALayer):
                modules_to_restore.append(name)
        
        if modules_to_restore:
            print(f" -> Merging {len(modules_to_restore)} Inverse PiSSA layers...")
            for name in modules_to_restore:
                if "." in name:
                    parent_name, child_name = name.rsplit(".", 1)
                    parent = model_merged.get_submodule(parent_name)
                else:
                    parent = model_merged
                    child_name = name
                
                pissa_layer = getattr(parent, child_name)
                # nn.Linear に戻す
                linear_layer = pissa_layer.merge_to_linear()
                setattr(parent, child_name, linear_layer)
                pissa_merged_count += 1
            print(f"    Merged {pissa_merged_count} layers.")
        
        # 3. LayerScale 適用
        print(" -> Merging LayerScale weights (Stabilization)...")
        layers = model_merged.model.layers
        with torch.no_grad():
            for i, layer in enumerate(layers):
                if hasattr(layer.self_attn, "o_proj"):
                    scale = self.scales[f"layer_{i}_attn"].item()
                    layer.self_attn.o_proj.weight.data *= scale
                if hasattr(layer.mlp, "down_proj"):
                    scale = self.scales[f"layer_{i}_mlp"].item()
                    layer.mlp.down_proj.weight.data *= scale
        
        print(f"Saving fully merged model to {output_dir}...")
        model_merged.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)


# --- 3. データセット処理 ---
def process_single_dataset(name, config_name, tokenizer, max_length=512):
    print(f"Loading: {name} (config: {config_name})")
    try:
        if name.endswith(".json") or name.endswith(".jsonl") or name.endswith(".csv"):
            ext = "json" if "json" in name else "csv"
            ds = load_dataset(ext, data_files=name, split="train")
        else:
            if config_name:
                ds = load_dataset(name, config_name, split="train")
            else:
                ds = load_dataset(name, split="train")
    except Exception as e:
        print(f"Failed to load {name}: {e}")
        return None
    cols = ds.column_names
    if "query" in cols and "answer" in cols:
        def tokenize_chat(examples):
            texts = []
            for q, a in zip(examples['query'], examples['answer']):
                messages = [{"role": "user", "content": q}, {"role": "assistant", "content": a}]
                try: text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
                except: text = f"user: {q}\nassistant: {a}\n</s>"
                texts.append(text)
            return tokenizer(texts, padding="max_length", truncation=True, max_length=max_length)
        ds = ds.map(tokenize_chat, batched=True, remove_columns=cols)
    elif "text" in cols:
        def tokenize_text(examples):
            texts = [t + tokenizer.eos_token for t in examples['text']]
            return tokenizer(texts, padding="max_length", truncation=True, max_length=max_length)
        ds = ds.map(tokenize_text, batched=True, remove_columns=cols)
    else:
        return None
    
    def add_labels(examples):
        examples["labels"] = examples["input_ids"].copy()
        return examples
    return ds.map(add_labels, batched=True)

def load_and_process_datasets(dataset_names, dataset_config, tokenizer, max_length=512):
    all_processed_datasets = []
    name_list = dataset_names.split(",")
    for name in name_list:
        name = name.strip()
        if not name: continue
        processed = process_single_dataset(name, dataset_config, tokenizer, max_length)
        if processed is not None: all_processed_datasets.append(processed)
    if not all_processed_datasets: raise ValueError("No valid datasets loaded.")
    return concatenate_datasets(all_processed_datasets)

# --- 4. メインスクリプト ---

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--dataset_names", type=str, required=True)
    parser.add_argument("--dataset_config", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="./output_stable_pissa")
    
    # Mode flags
    parser.add_argument("--only_train_layerscale", action="store_true")
    parser.add_argument("--use_lora", action="store_true")
    parser.add_argument("--use_inverse_pissa", action="store_true", help="Enable Inverse PiSSA (Tail-Tuning)")
    
    # Tuning Hyperparams
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--pissa_rank", type=int, default=128, help="Rank for Inverse PiSSA (Trainable tail components)")
    parser.add_argument("--pissa_target_modules", type=str, default="o_proj,down_proj", help="Comma separated modules for PiSSA")
    
    # Training Hyperparams
    parser.add_argument("--max_steps", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    
    # Stabilization Loss Params
    parser.add_argument("--lambda_lpf", type=float, default=0.05)
    parser.add_argument("--lambda_hpf", type=float, default=0.02)
    parser.add_argument("--lambda_anchor", type=float, default=0.05)
    parser.add_argument("--init_scale", type=float, default=0.2)
    
    
    # Logging
    parser.add_argument("--wandb", action="store_true", help="Enable logging to Weights & Biases.")
    parser.add_argument("--wandb_project", type=str, default="ipissa-liger-sft")
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--tensorboard", action="store_true")
    
    args = parser.parse_args()
    
    # トークナイザー
    print(f"Loading model: {args.model_id}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    
    # モデルロード
    # accelerateが設定するLOCAL_RANKから、このプロセスが使用すべきデバイスを取得
    # ローカルランクが設定されていない場合(シングルGPU実行)はcuda:0をフォールバックとして使用
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = f"cuda:{local_rank}"
    
    model = AutoLigerKernelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        attn_implementation="flash_attention_2",
        trust_remote_code=True,
        # device_map = "auto"
    ).to(device)
    model.gradient_checkpointing_enable()

    # --- 構造変更の適用順序 ---
    
    # 1. Inverse PiSSA (Tail-Tuning) の適用
    #    LoRAよりも先に構造を変える必要がある
    if args.use_inverse_pissa:
        target_modules = args.pissa_target_modules.split(",")
        model = apply_inverse_pissa(model, target_modules=target_modules, rank=args.pissa_rank)
        print(f"[Mode: Inverse PiSSA] Tunable rank: {args.pissa_rank}, Modules: {target_modules}")

    # 2. LoRA (PEFT) の適用
    #    Inverse PiSSAと併用する場合、PiSSA層以外の層(q_projなど)にLoRAをかけることも可能だが、
    #    通常は排他的、あるいは補完的に使う。
    #    PiSSA層は nn.Linear ではないため、PEFTは自動的にスキップするはずだが注意が必要。
    if args.use_lora and not args.use_inverse_pissa:
        print("[Mode: LoRA] Adding standard LoRA adapters...")
        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_r,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            task_type=TaskType.CAUSAL_LM,
            bias="none",
            lora_dropout=0.05,
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    # 3. 安定化ラッパーの適用 (LayerScale & Losses)
    #    PiSSA層に対しても forward hook をかけることで LayerScale は機能する
    print("Applying Stabilization Wrapper...")
    wrapped_model = StabilizedLigerWrapper(
        model, 
        init_scale=args.init_scale,
        lambda_lpf=args.lambda_lpf,
        lambda_hpf=args.lambda_hpf,
        lambda_anchor=args.lambda_anchor
    )

    # 4. 学習対象パラメータの設定
    if args.use_inverse_pissa:
        # PiSSAモードの場合:
        # - PiSSA層の lora_A, lora_B は nn.Parameter なのでデフォルトで requires_grad=True
        # - base_weight は buffer なので False
        # - LayerScale (scales) は True
        # 他の層を凍結する必要がある
        print("Freezing non-PiSSA/non-Scale parameters...")
        for name, param in wrapped_model.named_parameters():
            if "lora_" in name or "scales" in name:
                param.requires_grad = True
            elif "bias" in name and param.requires_grad: 
                # バイアス学習はお好みで
                param.requires_grad = True
            else:
                param.requires_grad = False
                
        # 確認表示
        trainable_params = sum(p.numel() for p in wrapped_model.parameters() if p.requires_grad)
        all_params = sum(p.numel() for p in wrapped_model.parameters())
        print(f"Trainable params: {trainable_params:,} / {all_params:,} ({100*trainable_params/all_params:.2f}%)")

    # データセットロード
    train_dataset = load_and_process_datasets(args.dataset_names, args.dataset_config, tokenizer, args.max_length)
    
    # Logging
    reporters = []
    if args.tensorboard:
        reporters.append("tensorboard")
    if args.wandb:
        reporters.append("wandb")
        
        if args.wandb_run_name:
            os.environ["WANDB_PROJECT"] = args.wandb_project
        if args.wandb_run_name:
            os.environ["WANDB_RUN_NAME"] = args.wandb_run_name

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        logging_steps=10,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        save_strategy="steps",
        save_steps=50,
        save_total_limit=2,
        report_to=reporters,
        remove_unused_columns=False,
        lr_scheduler_type="constant",   # schedule_free_radamなら問題ない
        optim="schedule_free_radam"
    )

    trainer = Trainer(
        model=wrapped_model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )

    print("Starting training...")
    trainer.train()

    print("Saving merged model...")
    wrapped_model.save_merged_model(args.output_dir, tokenizer)
    print("Done!")

if __name__ == "__main__":
    train()