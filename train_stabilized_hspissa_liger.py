import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    TokenizersBackend,
    SentencePieceBackend
)
from peft import LoraConfig, TaskType, get_peft_model
from peft.tuners.lora import Linear as LoraLinear
from liger_kernel.transformers import AutoLigerKernelForCausalLM
from datasets import load_dataset, load_dataset_builder, concatenate_datasets
import schedulefree
import argparse
import os
import types
import gc
from typing import Any

# --- 1. Head Skipped PiSSA (Tail-Tuning) 実装 ---

class HeadSkippedPiSSALayer(nn.Module):
    """
    Head Skipped PiSSA: 主要成分(Top Singular Values)上位を固定し、
    その下のrank分のみを学習する層。
    """
    def __init__(self, original_linear, rank=128, alpha=1.0, rank_skip=1):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.rank_skip = rank_skip
        self.in_features = original_linear.in_features
        self.out_features = original_linear.out_features
        self.dtype = original_linear.weight.dtype
        
        self.initialized = False
        
        # 元の重みを一時的にバッファとして保持
        # DDP broadcast/load_state_dict で「無いとエラー」にならないよう persistent=False
        # (SVD初期化のために一時的に参照できれば十分で、チェックポイントに含めない)
        self.register_buffer('original_weight', original_linear.weight.data.clone(), persistent=False)
        
         # 学習対象のパラメータと凍結部分のバッファを未初期化状態で確保
        # self.lora_A = nn.Parameter(torch.empty(self.out_features, self.rank, dtype=self.dtype))
        # self.lora_B = nn.Parameter(torch.empty(self.rank, self.in_features, dtype=self.dtype))
        self.lora_A_delta = nn.Parameter(torch.empty(self.out_features, self.rank, dtype=self.dtype))
        self.lora_B_delta = nn.Parameter(torch.empty(self.rank, self.in_features, dtype=self.dtype))
        self.register_buffer('lora_A_fixed', torch.empty(self.out_features, self.rank, dtype=self.dtype))
        self.register_buffer('lora_B_fixed', torch.empty(self.rank, self.in_features, dtype=self.dtype))
        self.register_buffer('weight_base', torch.empty(self.out_features, self.in_features, dtype=self.dtype))
        
        if original_linear.bias is not None:
            self.bias = nn.Parameter(original_linear.bias.data.clone())
        else:
            self.register_parameter('bias', None)
        
    def perform_svd_and_initialize(self, svd_device='cpu'):
        """
        保持している元の重みデータを使用してSVDを実行し、パラメータを初期化。
        マルチGPU環境では、モデルを各GPUに転送する前に、
        CPU上のモデルに対してこのメソッドを呼び出すことを想定。
        
        Args:
            svd_device (str or torch.device): SVD計算を実行するデバイス。
        """
        if self.initialized:
            print("  > Warning: Already initialized. Skipping SVD.")
            return

        if not hasattr(self, 'original_weight') or self.original_weight is None:
            raise RuntimeError("Original weight not found. It may have been already used and deleted.")

        # このモジュールが最終的に配置されるデバイス
        target_device = self.lora_A_delta.device
        
        # SVD計算のために、元の重みを指定されたデバイスのfloat32に転送
        W = self.original_weight.to(device=svd_device, dtype=torch.float32)
        
        print(f"  > SVD computing for shape {W.shape} on {svd_device}...")
        
        # 指定されたデバイスでSVDを実行
        U, S, Vh = torch.linalg.svd(W, full_matrices=False)

        # 2. 成分分離 (計算は引き続き svd_device で行われる)
        keep_indices_top = slice(0, self.rank_skip)
        keep_indices_bottom = slice(self.rank_skip + self.rank, None)
        train_indices = slice(self.rank_skip, self.rank_skip + self.rank)

        # W_base = U[:, keep_indices] @ torch.diag(S[keep_indices]) @ Vh[keep_indices, :]
        # W_base = torch.cat(
        #     [U[:, keep_indices_top], torch.zeros_like(U[:, train_indices]), U[:, keep_indices_bottom]], dim=1) \
        #     @ torch.diag(
        #         torch.cat([S[keep_indices_top], torch.zeros_like(S[train_indices]), S[keep_indices_bottom]], dim=0)) \
        #     @ torch.cat(
        #         [Vh[keep_indices_top, :], torch.zeros_like(Vh[train_indices, :]), Vh[keep_indices_bottom, :]], dim=0)
        W_base = torch.cat([U[:, keep_indices_top], U[:, keep_indices_bottom]], dim=1) @ \
            torch.diag(torch.cat([S[keep_indices_top], S[keep_indices_bottom]], dim=0)) @ \
            torch.cat([Vh[keep_indices_top, :], Vh[keep_indices_bottom, :]], dim=0)
        
        U_tail = U[:, train_indices]
        S_tail = S[train_indices]
        Vh_tail = Vh[train_indices, :]
        
        S_sqrt = torch.diag(torch.sqrt(S_tail))
        A_init = U_tail @ S_sqrt
        B_init = S_sqrt @ Vh_tail

        # 3. パラメータ登録
        # 計算結果をこのモジュールの本来のデバイス(target_device)と型に戻してコピー
        with torch.no_grad():
            self.weight_base.copy_(W_base.to(device=target_device, dtype=self.dtype))
            # self.lora_A.copy_(A_init.to(device=target_device, dtype=self.dtype))
            # self.lora_B.copy_(B_init.to(device=target_device, dtype=self.dtype))
            self.lora_A_fixed.copy_(A_init.to(device=target_device, dtype=self.dtype))
            self.lora_B_fixed.copy_(B_init.to(device=target_device, dtype=self.dtype))
            
            # 学習用deltaは0初期化
            self.lora_A_delta.zero_()
            self.lora_B_delta.zero_()
        
        self.initialized = True
        
        # メモリを解放するために一時的な重みを破棄
        self.original_weight = None
        
        # SVD計算デバイス上のテンソルを解放
        del W, U, S, Vh, W_base, A_init, B_init
        if torch.device(svd_device).type == 'cuda':
            torch.cuda.synchronize(svd_device)
            torch.cuda.empty_cache()

    def forward(self, x):
        if not self.initialized:
            raise RuntimeError(
                "HeadSkippedPiSSALayer is not initialized. "
                "Call perform_svd_and_initialize() on the CPU model before moving to the GPU."
            )
            
        # Base (Frozen) path
        base_out = nn.functional.linear(x, self.weight_base, self.bias)
        # # Adapter (Trainable Tail) path: x @ B.T @ A.T
        # adapter_out = (x @ self.lora_B.T) @ self.lora_A.T
        
        # Adapter (Fixed SVD + Learnable Delta)
        # (A_fixed + A_delta) @ (B_fixed + B_delta)
        # 初期状態では Delta=0 なので A_fixed @ B_fixed (元のTail) となる
        A_eff = self.lora_A_fixed + self.lora_A_delta
        B_eff = self.lora_B_fixed + self.lora_B_delta
        
        adapter_out = (x @ B_eff.T) @ A_eff.T
        
        return base_out + (self.alpha * adapter_out)

    def merge_to_linear(self):
        """標準的なnn.Linearに戻す"""
        if not self.initialized:
            raise RuntimeError("HeadSkippedPiSSALayer is not initialized.")
            
        with torch.no_grad():
            # W_new = self.weight_base + (self.lora_A @ self.lora_B)
            
            A_eff = self.lora_A_fixed + self.lora_A_delta
            B_eff = self.lora_B_fixed + self.lora_B_delta
            W_new = self.weight_base + (A_eff @ B_eff)
            
            new_linear = nn.Linear(
                in_features=self.in_features,
                out_features=self.out_features,
                bias=(self.bias is not None),
                device=W_new.device, # 現在のデバイスに作成
                dtype=W_new.dtype   # 現在のデータ型に合わせる
            )
            new_linear.weight.data.copy_(W_new)
            if self.bias is not None:
                new_linear.bias.data.copy_(self.bias)
            return new_linear

def apply_inverse_pissa(model, target_modules=["o_proj", "down_proj"], rank=128, rank_skip=1):
    """モデル内の指定層をHeadSkippedPiSSALayerに置換"""
    print(f"Converting target modules {target_modules} to Inverse PiSSA (Rank {rank}, Skipped head components {rank_skip})...")
    
    # 再帰的にモジュールを探索して置換
    # named_modules()だと置換中にイテレータが壊れる可能性があるため、名前リストを先に作る
    modules_to_replace = []
    for name, module in model.named_modules():
        if any(t in name for t in target_modules) and isinstance(module, (nn.Linear, LoraLinear)): 
            # LigerのLinearやLoRA層も対象にする場合、ここを調整
            # LigerKernelForCausalLMの場合、Linearは通常のnn.Linearではない可能性があるが、
            # .weight属性があれば動作するようにHeadSkippedPiSSALayerを作ってある
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
        
        # 新しい層を作成
        pissa_layer = HeadSkippedPiSSALayer(module, rank=rank, rank_skip=rank_skip)
        
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
            if hasattr(model.model.layers, "layers"):
                # MoE Case
                self.layers = model.model.layers.layers
            else:
                self.layers = model.model.layers
        elif hasattr(model, "base_model") and hasattr(model.base_model.model, "model"): # LoRA Case
            self.layers = model.base_model.model.model.layers
        else: # Generic Fallback
            self.layers = model.model.layers
            
        self.num_layers = len(self.layers)

        ref_param = next(self.model.parameters(), None)
        ref_device = ref_param.device if ref_param is not None else torch.device("cpu")
        ref_dtype = ref_param.dtype if ref_param is not None else torch.float32
        
        for i in range(self.num_layers):
            self.scales[f"layer_{i}_attn"] = nn.Parameter(torch.ones(1, device=ref_device, dtype=ref_dtype) * init_scale)
            self.scales[f"layer_{i}_mlp"] = nn.Parameter(torch.ones(1, device=ref_device, dtype=ref_dtype) * init_scale)

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
            if hasattr(layer, "mlp") and hasattr(layer.mlp, "down_proj"):
                original_down_proj_forward = layer.mlp.down_proj.forward
                key_mlp = f"layer_{i}_mlp"

                def new_down_proj_forward(module_self, x, *args, wrapper_ref=self, param_key=key_mlp, orig_f=original_down_proj_forward, **kwargs):
                    out = orig_f(x, *args, **kwargs)
                    scale = wrapper_ref.scales[param_key]
                    return out * scale

                layer.mlp.down_proj.forward = types.MethodType(new_down_proj_forward, layer.mlp.down_proj)
                
            moe_block = getattr(layer, "block_sparse_moe", getattr(layer, "mlp", None))
            
            if moe_block is not None and hasattr(moe_block, "experts"):
                # Expertsのリストをループ処理
                for expert_idx, expert in enumerate(moe_block.experts):
                    # Expert内の出力層を探す (Mixtralは 'w2', Qwen/Llamaは 'down_proj')
                    target_linear = getattr(expert, "down_proj", getattr(expert, "w2", None))
                    
                    if target_linear is not None:
                        # パラメータ管理用のキーを一意にする
                        key_moe = f"layer_{i}_moe_exp_{expert_idx}"
                        
                        # Scaleパラメータが未登録なら登録 (初期化)
                        if key_moe not in self.scales:
                            ref_param = next(self.model.parameters())
                            self.scales[key_moe] = nn.Parameter(
                                torch.ones(1, device=ref_param.device, dtype=ref_param.dtype) * 0.1 # init_scale
                            )

                        original_forward = target_linear.forward

                        def new_moe_forward(module_self, x, *args, wrapper_ref=self, param_key=key_moe, orig_f=original_forward, **kwargs):
                            out = orig_f(x, *args, **kwargs)
                            scale = wrapper_ref.scales[param_key]
                            return out * scale

                        target_linear.forward = types.MethodType(new_moe_forward, target_linear)
                
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
        # モデル内の HeadSkippedPiSSALayer を探して置換する
        pissa_merged_count = 0
        modules_to_restore = []
        for name, module in model_merged.named_modules():
            if isinstance(module, HeadSkippedPiSSALayer):
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
        if hasattr(model_merged.model.layers, "layers"):
            layers = model_merged.model.layers.layers
        else:
            layers = model_merged.model.layers
        with torch.no_grad():
            for i, layer in enumerate(layers):
                if hasattr(layer.self_attn, "o_proj"):
                    scale = self.scales[f"layer_{i}_attn"].item()
                    print(f"    Applying LayerScale to layer {i} attn: {scale}")
                    layer.self_attn.o_proj.weight.data *= scale
                if hasattr(layer, "mlp") and hasattr(layer.mlp, "down_proj"):
                    scale = self.scales[f"layer_{i}_mlp"].item()
                    print(f"    Applying LayerScale to layer {i} mlp: {scale}")
                    layer.mlp.down_proj.weight.data *= scale
                    
                moe_block = getattr(layer, "block_sparse_moe", getattr(layer, "mlp", None))
                    
                if moe_block is not None and hasattr(moe_block, "experts"):
                    # Expertsのリストをループ処理
                    for expert_idx, expert in enumerate(moe_block.experts):
                        # Expert内の出力層を探す (Mixtralは 'w2', Qwen/Llamaは 'down_proj')
                        target_linear = getattr(expert, "down_proj", getattr(expert, "w2", None))
                        
                        if target_linear is not None:
                            key_moe = f"layer_{i}_moe_exp_{expert_idx}"
                            scale = self.scales[key_moe].item()
                            print(f"    Applying LayerScale to layer {i} MoE expert {expert_idx}: {scale}")
                            target_linear.weight.data *= scale
        
        print(f"Saving fully merged model to {output_dir}...")
        model_merged.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)


# --- 3. データセット処理 ---
def process_single_dataset(name, config_name, tokenizer, max_length=512, add_token_str=None):
    print(f"Loading: {name} (config: {config_name})")
    try:
        if name.endswith(".json") or name.endswith(".jsonl") or name.endswith(".csv"):
            ext = "json" if "json" in name else "csv"
            ds = load_dataset(ext, data_files=name, split="train")
        else:
            cfg = None if config_name in (None, "", "None") else config_name
            if cfg is not None:
                ds = load_dataset(name, cfg, split="train")
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
                try:
                    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
                except:
                    text = f"user: {q}\nassistant: {a}\n</s>"
                if add_token_str is not None:
                    text = add_token_str + text
                texts.append(text)
            return tokenizer(texts, padding="max_length", truncation=True, max_length=max_length)
        ds = ds.map(tokenize_chat, batched=True, remove_columns=cols)
    elif "text" in cols:
        def tokenize_text(examples):
            add_str = "" if add_token_str is None else add_token_str
            texts = [add_str + t + tokenizer.eos_token for t in examples['text']]
            return tokenizer(texts, padding="max_length", truncation=True, max_length=max_length)
        ds = ds.map(tokenize_text, batched=True, remove_columns=cols)
    else:
        return None
    
    def add_labels(examples):
        examples["labels"] = examples["input_ids"].copy()
        return examples
    return ds.map(add_labels, batched=True)

def load_and_process_datasets(dataset_names, dataset_config, tokenizer, max_length=512, add_first_tokens=None, local_rank=0, procecced_callback=None):
    all_processed_datasets = []
    name_list = dataset_names.split(",")
    if dataset_config is not None:
        dataset_config = dataset_config.split(",")
    elif len(name_list) > 1:
        dataset_config = [None] * len(name_list)
    if add_first_tokens is not None:
        add_token_str = add_first_tokens.split(",")
    else:
        add_token_str = [None] * len(name_list)
        
    if local_rank == 0:
        # トークナイズや結合を先に行っておくため、
        # local_rankが0のプロセスのみに処理させる
        for name, config, add_token in zip(name_list, dataset_config, add_token_str):
            name = name.strip()
            if not name:
                continue
            processed = process_single_dataset(name, config, tokenizer, max_length, add_token)
            if processed is not None:
                all_processed_datasets.append(processed)
        if not all_processed_datasets:
            raise ValueError("No valid datasets loaded.")
        concatenated_dataset = concatenate_datasets(all_processed_datasets)
    if procecced_callback is not None:
        # 同期用
        procecced_callback()
    
    # データセットを全プロセスでロード
    for name, config, add_token in zip(name_list, dataset_config, add_token_str):
        name = name.strip()
        if not name:
            continue
        processed = process_single_dataset(name, config, tokenizer, max_length, add_token)
        if processed is not None:
            all_processed_datasets.append(processed)
    if not all_processed_datasets:
        raise ValueError("No valid datasets loaded.")
    concatenated_dataset = concatenate_datasets(all_processed_datasets)
    
    return concatenated_dataset

# --- 4. メインスクリプト ---

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--tokenizer_id", type=str, default=None, help="If empty, uses the same model as the base model.")
    parser.add_argument("--dataset_names", "--dataset_name", dest="dataset_names", type=str, required=True)
    parser.add_argument("--dataset_config", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="./output_stable_pissa")
    
    parser.add_argument("--add_first_tokens", type=str, default=None, help="Comma separated list of additional tokens to add to the beginning of the input sequence.")
    
    # Mode flags
    parser.add_argument("--only_train_layerscale", action="store_true")
    parser.add_argument("--use_lora", action="store_true")
    parser.add_argument("--use_head_skipped_pissa", action="store_true", help="Enable Head Skipped PiSSA (Tail-Tuning)")
    
    parser.add_argument("--use_ddp",  action="store_true", help="Enable Distributed Data Parallel (DDP)")
    parser.add_argument("--svd_device", type=str, default="cpu", help="SVD device")
    
    parser.add_argument("--optim_setup", type=str, default="schedule_free_radam", help="optimizer setup")
    
    parser.add_argument("--attn_implementation", type=str, default="flash_attention_2", help="Attention implementation")
    parser.add_argument("--use_liger_kernel", action="store_true", help="Use LigerKernelForCausalLM instead of AutoModelForCausalLM")
    
    # Tuning Hyperparams
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--pissa_rank", type=int, default=128, help="Rank for Head Skipped PiSSA (Trainable tail components)")
    parser.add_argument("--pissa_rank_skip", type=int, default=1, help="Rank to skip for Head Skipped PiSSA (Num skipped head components)")
    parser.add_argument("--pissa_target_modules", type=str, default="o_proj,down_proj", help="Comma separated modules for PiSSA")
    
    # Training Hyperparams
    parser.add_argument("--max_steps", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--dataset_seed", type=int, default=3303)
    
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
    
    
    accelerator = None
    broadcast_object_list = None
    if args.use_ddp:
        from accelerate import Accelerator
        from accelerate.utils import broadcast_object_list
        accelerator = Accelerator()
    
    # トークナイザー
    tokenizer_id = args.tokenizer_id if args.tokenizer_id is not None else args.model_id
    print(f"Loading tokenizer model: {tokenizer_id}")
    tokenizer: TokenizersBackend | SentencePieceBackend = AutoTokenizer.from_pretrained(
        tokenizer_id,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    
    add_first_tokens = None
    num_added_tokens = 0
    if args.add_first_tokens is not None:
        # prefixトークンの追加処理
        add_first_tokens = args.add_first_tokens.split(",")
        for token in add_first_tokens:
            if len(token) > 0 and len(tokenizer.tokenize(token)) > 1:
                # 新規トークンとして追加
                tokenizer.add_tokens([token])
                num_added_tokens += 1
    if num_added_tokens > 0:
        print(f"Added {num_added_tokens} tokens to the tokenizer.")
    
    # モデルロード
    # accelerateが設定するLOCAL_RANKから、このプロセスが使用すべきデバイスを取得
    # ローカルランクが設定されていない場合(シングルGPU実行)はcuda:0をフォールバックとして使用
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = f"cuda:{local_rank}"
    
    # 一旦モデルロード
    if accelerator is not None:
        main_process_ctx = accelerator.main_process_first()
    else:
        from contextlib import nullcontext
        main_process_ctx = nullcontext()

    with main_process_ctx:
        autoModelClass = AutoLigerKernelForCausalLM if args.use_liger_kernel else AutoModelForCausalLM
        model = autoModelClass.from_pretrained(
            args.model_id,
            torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            attn_implementation=args.attn_implementation,
            trust_remote_code=True,
            # device_map = "auto"
        )
        model_dtype = model.dtype
        
        if num_added_tokens > 0:
            # embed_tokensのりサイズ
            model.resize_token_embeddings(len(tokenizer))

    # --- 構造変更の適用 ---
    
    # 1. Head Skipped PiSSA (Tail-Tuning) の適用
    #    LoRAよりも先に構造を変える必要がある
    if args.use_head_skipped_pissa:
        target_modules = args.pissa_target_modules.split(",")
        model = apply_inverse_pissa(model, target_modules=target_modules, rank=args.pissa_rank, rank_skip=args.pissa_rank_skip)
        print(f"[Mode: Head Skipped PiSSA] Tunable rank: {args.pissa_rank}, Modules: {target_modules}")

        object_to_broadcast = None
        if int(local_rank) == 0:
            # rank0のみが変換実行
            
            # モデル全体をSVD計算用のデバイスに一旦移動
            model.to(device=args.svd_device)
            # 指定したデバイス上でSVDを実行し、パラメータを初期化
            print(f"Initializing HeadSkippedPiSSALayer parameters on {args.svd_device}...")
            model.apply(lambda module:
                module.perform_svd_and_initialize(svd_device=args.svd_device)
                if isinstance(module, HeadSkippedPiSSALayer) else None
            )
            
            if args.use_ddp:
                # state_dictをCPUで取得
                state_dict = model.cpu().state_dict()
                
                # ブロードキャストするオブジェクトを準備
                object_to_broadcast = [state_dict]
        elif args.use_ddp:
            # rank0以外は受信用のプレースホルダを作成
            object_to_broadcast = [None]
            
        object_to_broadcast: list[Any]
        
        if args.use_ddp:
            # ブロードキャスト実行
            broadcast_object_list(object_to_broadcast, from_process=0)
            
            # 受信したモデルをロード
            print(f"--- Process {accelerator.process_index}: Loading broadcasted state_dict ---")
            model.load_state_dict(object_to_broadcast[0], strict=False)

            # broadcastではPython属性(initialized)は同期されないため、全rankで揃える
            def _finalize_inverse_pissa(module):
                if isinstance(module, HeadSkippedPiSSALayer):
                    module.initialized = True
                    module.original_weight = None
            model.apply(_finalize_inverse_pissa)
            
            # メモリを解放
            del object_to_broadcast
            
            # 一応ブロック
            accelerator.wait_for_everyone()

    # 2. LoRA (PEFT) の適用
    #    Inverse PiSSAと併用する場合、PiSSA層以外の層(q_projなど)にLoRAをかけることも可能だが、
    #    通常は排他的、あるいは補完的に使う。
    #    PiSSA層は nn.Linear ではないため、PEFTは自動的にスキップするはずだが注意が必要。
    if args.use_lora and not args.use_head_skipped_pissa:
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
        
    # 念の為モデルの型を変更
    model = model.to(dtype=model_dtype)
    print(f"Model dtype: {model.dtype}")

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
    if args.use_head_skipped_pissa:
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
    train_dataset = load_and_process_datasets(args.dataset_names, args.dataset_config, tokenizer, args.max_length, args.add_first_tokens, local_rank, None if not args.use_ddp else accelerator.wait_for_everyone)
    # Shuffle
    train_dataset = train_dataset.shuffle(seed=args.dataset_seed)
    
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
            
    optim_setup = {}
    if args.optim_setup == "schedule_free_radam":
        optim_setup["lr_scheduler_type"] = "constant"
        optim_setup["optim"] = "schedule_free_radam"
    elif args.optim_setup == "adamw_torch_fused":
        optim_setup["lr_scheduler_type"] = "cosine"
        optim_setup["optim"] = "adamw_torch_fused"
        optim_setup["warmup_steps"] = 100
    else:
        raise ValueError(f"Unknown optimizer setup: {args.optim_setup}")

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
        save_steps=100,
        save_total_limit=2,
        report_to=reporters,
        remove_unused_columns=False,
        # lr_scheduler_type="constant",   # schedule_free_radamなら問題ない
        # optim="schedule_free_radam"
        # optim="adamw_torch_fused",
        # lr_scheduler_type="cosine",
        # warmup_steps=100,
        **optim_setup
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