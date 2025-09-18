import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import types
from typing import List, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import (
    get_peft_model,
    prepare_model_for_kbit_training,
    LoraConfig,
    AdaptionPromptConfig,
    PrefixTuningConfig,
)
from loguru import logger
from model.config import ModelConfig, TrainConfig
from omegaconf import OmegaConf


def compute_accuracy(pad_outputs, pad_targets, ignore_label):
    """Calculate accuracy.

    Args:
        pad_outputs (LongTensor): Prediction tensors (B, Lmax).
        pad_targets (LongTensor): Target label tensors (B, Lmax).
        ignore_label (int): Ignore label id.

    Returns:
        float: Accuracy value (0.0 - 1.0).

    """
    mask = pad_targets != ignore_label
    numerator = torch.sum(
        pad_outputs.masked_select(mask) == pad_targets.masked_select(mask)
    )
    denominator = torch.sum(mask)
    return numerator.float() / denominator.float()


def generate_peft_config(train_config):
    peft_configs = {
        "lora": LoraConfig,
        "llama_adapter": AdaptionPromptConfig,
        "prefix": PrefixTuningConfig,
    }
    config = train_config.peft_config
    params = OmegaConf.to_container(config, resolve=True)
    params.pop("peft_method", None)
    peft_config = peft_configs[config.get("peft_method", "lora")](**params)
    return peft_config


def print_model_size(model, config, rank: int = 0) -> None:
    """
    log model name, the number of trainable parameters and initialization time.

    Args:
        model: The PyTorch model.
        model_name (str): Name of the model.
        rank (int, optional): Current process's rank. Defaults to 0.
    """
    if rank == 0:
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(
            f"Model [{config.model_name}] has {total_params / 1e6:,.2f} M prams"
        )


def print_module_size(module, module_name, rank: int = 0) -> None:
    """
    All params or Train params
    Print module name, the number of trainable parameters and initialization time.

    Args:
        module: The PyTorch module.
        module_name (str): Name of the model.
        rank (int, optional): Current process's rank. Defaults to 0.
    """
    if rank == 0:
        total_params = sum(p.numel() for p in module.parameters())
        train_params = sum(p.numel() for p in module.parameters() if p.requires_grad)

        logger.info(
            f"Module [{module_name}] has {train_params / 1e6:,.2f}/{total_params / 1e6:,.2f} M params"
        )


def setup_tokenizer(train_config: TrainConfig, model_config: ModelConfig, **kwargs):
    # Load the tokenizer and add special tokens
    tokenizer = AutoTokenizer.from_pretrained(model_config.llm_path)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer


def setup_encoder(train_config: TrainConfig, model_config: ModelConfig, **kwargs):
    encoder_name = model_config.encoder_name

    encoder = ConformerEncoder(**model_config["encoder_config"])

    if train_config.freeze_encoder:
        for name, param in encoder.named_parameters():
            param.requires_grad = False
        encoder.eval()
    print_module_size(
        encoder,
        encoder_name,
        int(os.environ["RANK"])
        if train_config.enable_fsdp or train_config.enable_ddp
        else 0,
    )

    return encoder


def setup_encoder_projector(
    train_config: TrainConfig, model_config: ModelConfig, **kwargs
):
    encoder_projector = Adapter(
        model_config["encoder_config"]["d_model"],
        model_config["llm_dim"],
        model_config["encoder_projector_ds_rate"],
    )

    return encoder_projector


def setup_llm(train_config: TrainConfig, model_config: ModelConfig, **kwargs):
    use_cache = False if train_config.enable_fsdp or train_config.enable_ddp else None

    model = AutoModelForCausalLM.from_pretrained(
        model_config.llm_path,
        # load_in_8bit=True if train_config.quantization else None,
        # device_map="auto" if train_config.quantization else None,
        use_cache=use_cache,
    )

    # Prepare the model for int8 training if quantization is enabled
    if train_config.quantization:
        model = prepare_model_for_kbit_training(model)

    if (
        train_config.freeze_llm
    ):  # TODO:to test offical `freeze_layers` and `num_freeze_layers`
        for name, param in model.named_parameters():
            param.requires_grad = False
        model.eval()

    if train_config.use_peft:
        peft_config = generate_peft_config(train_config)
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    if train_config.train_embed:
        model.get_input_embeddings().weight.requires_grad = True
        logger.info("Train embedding")
    if train_config.train_lm_head:
        out = model.get_output_embeddings()
        if out is not None:
            out.weight.requires_grad = True
            logger.info("Train lm head")

    print_module_size(
        model,
        model_config.llm_name,
        int(os.environ["RANK"])
        if train_config.enable_fsdp or train_config.enable_ddp
        else 0,
    )
    return model


def model_factory(train_config: TrainConfig, model_config: ModelConfig, **kwargs):
    tokenizer = setup_tokenizer(train_config, model_config, **kwargs)
    DEFAULT_SPEECH_TOKEN = "<speech>"
    DEFAULT_IGNORE_TOKEN = -100
    special_tokens_dict = {"additional_special_tokens": [DEFAULT_SPEECH_TOKEN]}
    tokenizer.add_special_tokens(special_tokens_dict)
    tokenizer.default_ignore_token = DEFAULT_IGNORE_TOKEN
    tokenizer.default_speech_token = tokenizer.convert_tokens_to_ids(
        DEFAULT_SPEECH_TOKEN
    )

    # llm
    llm = setup_llm(train_config, model_config, **kwargs)

    # encoder
    encoder = setup_encoder(train_config, model_config, **kwargs)

    # projector
    # train projector or not
    encoder_projector = setup_encoder_projector(train_config, model_config, **kwargs)

    model = TsAsr(
        encoder,
        llm,
        encoder_projector,
        tokenizer,
        train_config,
        model_config,
        **kwargs,
    )

    # load firered ckpt
    firered_path = model_config["firered_path"]
    firered_dict = torch.load(firered_path, map_location="cpu")
    model.load_state_dict(firered_dict["model_state_dict"], strict=False)
    ckpt_path = kwargs.get("ckpt_path", None)

    # customer train ckpt
    if ckpt_path is not None:
        logger.info("loading other parts from: {}".format(ckpt_path))
        ckpt_dict = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(ckpt_dict, strict=False)

    # train projector or not
    if train_config.freeze_projector:
        for name, param in model.encoder_projector.named_parameters():
            param.requires_grad = False
    else:
        for name, param in model.encoder_projector.named_parameters():
            param.requires_grad = True
    print_module_size(
        encoder_projector,
        "encoder_projector",
        int(os.environ["RANK"])
        if train_config.enable_fsdp or train_config.enable_ddp
        else 0,
    )
    # breakpoint()
    print_model_size(
        model,
        train_config,
        (
            int(os.environ["RANK"])
            if train_config.enable_fsdp or train_config.enable_ddp
            else 0
        ),
    )
    return model, tokenizer


class ConformerEncoder(nn.Module):
    def __init__(
        self,
        idim,
        n_layers,
        n_head,
        d_model,
        residual_dropout=0.1,
        dropout_rate=0.1,
        kernel_size=33,
        pe_maxlen=5000,
    ):
        super().__init__()
        self.odim = d_model

        self.input_preprocessor = Conv2dSubsampling(idim, d_model)
        self.positional_encoding = RelPositionalEncoding(d_model)
        self.dropout = nn.Dropout(residual_dropout)

        self.layer_stack = nn.ModuleList()
        for l in range(n_layers):
            block = RelPosEmbConformerBlock(
                d_model, n_head, residual_dropout, dropout_rate, kernel_size
            )
            self.layer_stack.append(block)

    def forward(self, padded_input, input_lengths, pad=True):
        if pad:
            padded_input = F.pad(
                padded_input,
                (0, 0, 0, self.input_preprocessor.context - 1),
                "constant",
                0.0,
            )
        src_mask = self.padding_position_is_0(padded_input, input_lengths)

        embed_output, input_lengths, src_mask = self.input_preprocessor(
            padded_input, src_mask
        )
        enc_output = self.dropout(embed_output)

        pos_emb = self.dropout(self.positional_encoding(embed_output))

        enc_outputs = []
        for enc_layer in self.layer_stack:
            enc_output = enc_layer(
                enc_output, pos_emb, slf_attn_mask=src_mask, pad_mask=src_mask
            )
            enc_outputs.append(enc_output)

        return enc_output, input_lengths, src_mask

    def padding_position_is_0(self, padded_input, input_lengths):
        N, T = padded_input.size()[:2]
        mask = torch.ones((N, T)).to(padded_input.device)
        for i in range(N):
            mask[i, input_lengths[i] :] = 0
        mask = mask.unsqueeze(dim=1)
        return mask.to(torch.uint8)


class RelPosEmbConformerBlock(nn.Module):
    def __init__(
        self, d_model, n_head, residual_dropout=0.1, dropout_rate=0.1, kernel_size=33
    ):
        super().__init__()
        self.ffn1 = ConformerFeedForward(d_model, dropout_rate)
        self.mhsa = RelPosMultiHeadAttention(n_head, d_model, residual_dropout)
        self.conv = ConformerConvolution(d_model, kernel_size, dropout_rate)
        self.ffn2 = ConformerFeedForward(d_model, dropout_rate)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x, pos_emb, slf_attn_mask=None, pad_mask=None):
        out = 0.5 * x + 0.5 * self.ffn1(x)
        out = self.mhsa(out, out, out, pos_emb, mask=slf_attn_mask)[0]
        out = self.conv(out, pad_mask)
        out = 0.5 * out + 0.5 * self.ffn2(out)
        out = self.layer_norm(out)
        return out


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class Conv2dSubsampling(nn.Module):
    def __init__(self, idim, d_model, out_channels=32):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, out_channels, 3, 2),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, 2),
            nn.ReLU(),
        )
        subsample_idim = ((idim - 1) // 2 - 1) // 2
        self.out = nn.Linear(out_channels * subsample_idim, d_model)

        self.subsampling = 4
        left_context = right_context = 3  # both exclude currect frame
        self.context = left_context + 1 + right_context  # 7

    def forward(self, x, x_mask):
        x = x.unsqueeze(1)
        x = self.conv(x)
        N, C, T, D = x.size()
        x = self.out(x.transpose(1, 2).contiguous().view(N, T, C * D))
        mask = x_mask[:, :, :-2:2][:, :, :-2:2]
        input_lengths = mask[:, -1, :].sum(dim=-1)
        return x, input_lengths, mask


class RelPositionalEncoding(torch.nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe_positive = torch.zeros(max_len, d_model, requires_grad=False)
        pe_negative = torch.zeros(max_len, d_model, requires_grad=False)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * -(torch.log(torch.tensor(10000.0)).item() / d_model)
        )
        pe_positive[:, 0::2] = torch.sin(position * div_term)
        pe_positive[:, 1::2] = torch.cos(position * div_term)
        pe_negative[:, 0::2] = torch.sin(-1 * position * div_term)
        pe_negative[:, 1::2] = torch.cos(-1 * position * div_term)

        pe_positive = torch.flip(pe_positive, [0]).unsqueeze(0)
        pe_negative = pe_negative[1:].unsqueeze(0)
        pe = torch.cat([pe_positive, pe_negative], dim=1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # Tmax = 2 * max_len - 1
        Tmax, T = self.pe.size(1), x.size(1)
        pos_emb = self.pe[:, Tmax // 2 - T + 1 : Tmax // 2 + T].clone().detach()
        return pos_emb


class ConformerFeedForward(nn.Module):
    def __init__(self, d_model, dropout_rate=0.1):
        super().__init__()
        pre_layer_norm = nn.LayerNorm(d_model)
        linear_expand = nn.Linear(d_model, d_model * 4)
        nonlinear = Swish()
        dropout_pre = nn.Dropout(dropout_rate)
        linear_project = nn.Linear(d_model * 4, d_model)
        dropout_post = nn.Dropout(dropout_rate)
        self.net = nn.Sequential(
            pre_layer_norm,
            linear_expand,
            nonlinear,
            dropout_pre,
            linear_project,
            dropout_post,
        )

    def forward(self, x):
        residual = x
        output = self.net(x)
        output = output + residual
        return output


class ConformerConvolution(nn.Module):
    def __init__(self, d_model, kernel_size=33, dropout_rate=0.1):
        super().__init__()
        assert kernel_size % 2 == 1
        self.pre_layer_norm = nn.LayerNorm(d_model)
        self.pointwise_conv1 = nn.Conv1d(
            d_model, d_model * 4, kernel_size=1, bias=False
        )
        self.glu = F.glu
        self.padding = (kernel_size - 1) // 2
        self.depthwise_conv = nn.Conv1d(
            d_model * 2,
            d_model * 2,
            kernel_size,
            stride=1,
            padding=self.padding,
            groups=d_model * 2,
            bias=False,
        )
        self.batch_norm = nn.LayerNorm(d_model * 2)
        self.swish = Swish()
        self.pointwise_conv2 = nn.Conv1d(
            d_model * 2, d_model, kernel_size=1, bias=False
        )
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, mask=None):
        residual = x
        out = self.pre_layer_norm(x)
        out = out.transpose(1, 2)
        if mask is not None:
            out.masked_fill_(mask.ne(1), 0.0)
        out = self.pointwise_conv1(out)
        out = F.glu(out, dim=1)
        out = self.depthwise_conv(out)

        out = out.transpose(1, 2)
        out = self.swish(self.batch_norm(out))
        out = out.transpose(1, 2)

        out = self.dropout(self.pointwise_conv2(out))
        if mask is not None:
            out.masked_fill_(mask.ne(1), 0.0)
        out = out.transpose(1, 2)
        return out + residual


class EncoderMultiHeadAttention(nn.Module):
    def __init__(self, n_head, d_model, residual_dropout=0.1):
        super().__init__()
        assert d_model % n_head == 0
        self.n_head = n_head
        self.d_k = d_model // n_head
        self.d_v = self.d_k

        self.w_qs = nn.Linear(d_model, n_head * self.d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * self.d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * self.d_v, bias=False)

        self.layer_norm_q = nn.LayerNorm(d_model)
        self.layer_norm_k = nn.LayerNorm(d_model)
        self.layer_norm_v = nn.LayerNorm(d_model)

        self.attention = ScaledDotProductAttention(temperature=self.d_k**0.5)
        self.fc = nn.Linear(n_head * self.d_v, d_model, bias=False)
        self.dropout = nn.Dropout(residual_dropout)

    def forward(self, q, k, v, mask=None):
        sz_b, len_q = q.size(0), q.size(1)

        residual = q
        q, k, v = self.forward_qkv(q, k, v)

        output, attn = self.attention(q, k, v, mask=mask)

        output = self.forward_output(output, residual, sz_b, len_q)
        return output, attn

    def forward_qkv(self, q, k, v):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        q = self.layer_norm_q(q)
        k = self.layer_norm_k(k)
        v = self.layer_norm_v(v)

        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        return q, k, v

    def forward_output(self, output, residual, sz_b, len_q):
        output = output.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        fc_out = self.fc(output)
        output = self.dropout(fc_out)
        output = output + residual
        return output


class ScaledDotProductAttention(nn.Module):
    def __init__(self, temperature):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(0.0)
        self.INF = float("inf")

    def forward(self, q, k, v, mask=None):
        attn = torch.matmul(q, k.transpose(2, 3)) / self.temperature
        output, attn = self.forward_attention(attn, v, mask)
        return output, attn

    def forward_attention(self, attn, v, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
            mask = mask.eq(0)
            attn = attn.masked_fill(mask, -self.INF)
            attn = torch.softmax(attn, dim=-1).masked_fill(mask, 0.0)
        else:
            attn = torch.softmax(attn, dim=-1)

        d_attn = self.dropout(attn)
        output = torch.matmul(d_attn, v)

        return output, attn


class RelPosMultiHeadAttention(EncoderMultiHeadAttention):
    def __init__(self, n_head, d_model, residual_dropout=0.1):
        super().__init__(n_head, d_model, residual_dropout)
        d_k = d_model // n_head
        self.scale = 1.0 / (d_k**0.5)
        self.linear_pos = nn.Linear(d_model, n_head * d_k, bias=False)
        self.pos_bias_u = nn.Parameter(torch.FloatTensor(n_head, d_k))
        self.pos_bias_v = nn.Parameter(torch.FloatTensor(n_head, d_k))
        torch.nn.init.xavier_uniform_(self.pos_bias_u)
        torch.nn.init.xavier_uniform_(self.pos_bias_v)

    def _rel_shift(self, x):
        N, H, T1, T2 = x.size()
        zero_pad = torch.zeros((N, H, T1, 1), device=x.device, dtype=x.dtype)
        x_padded = torch.cat([zero_pad, x], dim=-1)

        x_padded = x_padded.view(N, H, T2 + 1, T1)
        x = x_padded[:, :, 1:].view_as(x)
        x = x[:, :, :, : x.size(-1) // 2 + 1]
        return x

    def forward(self, q, k, v, pos_emb, mask=None):
        sz_b, len_q = q.size(0), q.size(1)

        residual = q
        q, k, v = self.forward_qkv(q, k, v)

        q = q.transpose(1, 2)
        n_batch_pos = pos_emb.size(0)
        p = self.linear_pos(pos_emb).view(n_batch_pos, -1, self.n_head, self.d_k)
        p = p.transpose(1, 2)

        q_with_bias_u = (q + self.pos_bias_u).transpose(1, 2)
        q_with_bias_v = (q + self.pos_bias_v).transpose(1, 2)

        matrix_ac = torch.matmul(q_with_bias_u, k.transpose(-2, -1))

        matrix_bd = torch.matmul(q_with_bias_v, p.transpose(-2, -1))
        matrix_bd = self._rel_shift(matrix_bd)

        attn_scores = matrix_ac + matrix_bd
        attn_scores.mul_(self.scale)

        output, attn = self.attention.forward_attention(attn_scores, v, mask=mask)

        output = self.forward_output(output, residual, sz_b, len_q)
        return output, attn


class Adapter(nn.Module):
    def __init__(self, encoder_dim, llm_dim, downsample_rate=2):
        super().__init__()
        self.ds = downsample_rate
        self.linear1 = nn.Linear(encoder_dim * downsample_rate, llm_dim)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(llm_dim, llm_dim)

    def forward(self, x):
        batch_size, seq_len, feat_dim = x.size()
        num_frames_to_discard = seq_len % self.ds
        if num_frames_to_discard > 0:
            x = x[:, :-num_frames_to_discard, :]
        seq_len = x.size(1)

        x = x.contiguous()
        x = x.view(batch_size, seq_len // self.ds, feat_dim * self.ds)

        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)

        return x


class TsAsr(torch.nn.Module):
    def __init__(
        self,
        encoder,
        llm,
        encoder_projector,
        tokenizer,
        train_config,
        model_config,
        **kwargs,
    ):
        super().__init__()
        # modality encoder
        self.encoder = encoder

        # llm
        self.llm = llm

        # projector
        self.encoder_projector = encoder_projector

        # tokenizer
        self.tokenizer = tokenizer
        self.metric = kwargs.get("metric", "acc")

        self.train_config = train_config
        self.model_config = model_config

        if train_config.get("enable_deepspeed", False):

            def new_forward(self, input):
                output = F.layer_norm(
                    input.float(),
                    self.normalized_shape,
                    self.weight.float() if self.weight is not None else None,
                    self.bias.float() if self.bias is not None else None,
                    self.eps,
                )
                return output.type_as(input)

            for item in self.modules():
                if isinstance(item, nn.LayerNorm):
                    item.forward = types.MethodType(new_forward, item)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        input_features: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        input_feature_length: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        encoder_outs, encoder_feature_length, _ = self.encoder(
            input_features, input_feature_length
        )  # bs*seq*dim
        projector_outs = self.encoder_projector(encoder_outs)
        projector_feature_length = encoder_feature_length // self.encoder_projector.ds
        # print("\n","End",inputs_embeds, attention_mask, labels, position_ids,encoder_feature_length,projector_feature_length)
        inputs_embeds = self.llm.get_input_embeddings()(input_ids)
        inputs_embeds, attention_mask, labels, position_ids, _ = (
            self._merge_input_ids_with_audio_features(
                projector_outs,
                projector_feature_length,
                inputs_embeds,
                input_ids,
                attention_mask,
                labels,
            )
        )
        # print("\n","Before", projector_outs.shape, input_feature_length,encoder_feature_length,projector_feature_length)
        # exit(1)
        model_outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            position_ids=position_ids,
        )
        acc = -1
        if self.metric:
            with torch.no_grad():
                preds = torch.argmax(model_outputs.logits, -1)
                acc = compute_accuracy(
                    preds.detach()[:, :-1],
                    labels.detach()[:, 1:],
                    ignore_label=self.tokenizer.default_ignore_token,
                )

        return model_outputs, acc

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.LongTensor = None,
        input_features: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        input_feature_length: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        encoder_outs, encoder_feature_length, _ = self.encoder(
            input_features, input_feature_length
        )  # bs*seq*dim
        projector_outs = self.encoder_projector(encoder_outs)
        projector_feature_length = encoder_feature_length // self.encoder_projector.ds
        inputs_embeds = self.llm.get_input_embeddings()(input_ids)
        inputs_embeds, attention_mask, labels, position_ids, _ = (
            self._merge_input_ids_with_audio_features(
                projector_outs,
                projector_feature_length,
                inputs_embeds,
                input_ids,
                attention_mask,
                labels,
            )
        )
        model_outputs = self.llm.generate(
            inputs_embeds=inputs_embeds,
            max_new_tokens=kwargs.get("max_new_tokens", 200),
            num_beams=kwargs.get("num_beams", 3),
            do_sample=kwargs.get("do_sample", False),
            min_length=kwargs.get("min_length", 1),
            top_p=kwargs.get("top_p", 1.0),
            repetition_penalty=kwargs.get("repetition_penalty", 3.0),
            length_penalty=kwargs.get("length_penalty", 1.0),
            temperature=kwargs.get("temperature", 1.0),
            attention_mask=attention_mask,
            # position_ids=position_ids,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        return model_outputs

    @property
    def device(self):
        return self.llm.device

    def eval(self):
        for name, param in self.llm.named_parameters():
            param.requires_grad = False
        self.llm.eval()
        for name, param in self.encoder_projector.named_parameters():
            param.requires_grad = False
        self.encoder_projector.eval()
        for name, param in self.encoder.named_parameters():
            param.requires_grad = False
        self.encoder.eval()

    def _merge_input_ids_with_audio_features(
        self,
        audio_features,
        num_audio_tokens,
        inputs_embeds,
        input_ids,
        attention_mask,
        labels,
    ):
        """
        Merge input_ids with with audio features into final embeddings

        Args:
            audio_features (`torch.Tensor` of shape `(num_audios, max_audio_tokens, embed_dim)`):
                All audio vectors of all audios in the batch
            num_audio_tokens (`torch.LongTensor` of shape `(num_audios)`):
                The length of audio embeddings of each audio as stacked in `audio_features`
            inputs_embeds (`torch.Tensor` of shape `(batch_size, sequence_length, embed_dim)`):
                Token embeddings before merging with audio embeddings
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Input_ids of tokens, possibly filled with audio token
            attention_mask (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Mask to avoid performing attention on padding token indices.
            labels (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*)
                labels need to be recalculated to support training (if provided)
        Returns:
            final_embedding, final_attention_mask, final_labels, position_ids, final_input_ids

        Explanation:
            each audio has variable length embeddings, with length specified by num_audio_tokens
            audio_features is concatenation of all audio embed vectors
            task: fill each <|AUDIO|> with the correct number of audio embeddings
            Example:
                X (5 tokens), Y (3 tokens), Z (8 tokens)
                X, Y are in the same sequence (in-context learning)
            if right padding
                input_ids: [
                    a b c d e f X g h i j k Y l m
                    o p q r Z s t u v _ _ _ _ _ _
                ]
                input_ids should be: [
                    a b c d e f X X X X X g h i j k Y Y Y l m
                    o p q r Z Z Z Z Z Z Z Z s t u v _ _ _ _ _
                ]
                labels should be: [
                    a b c d e f _ _ _ _ _ g h i j k _ _ _ l m
                    o p q r _ _ _ _ _ _ _ _ s t u v _ _ _ _ _
                ]
            elif left padding
                input_ids: [
                    a b c d e f X g h i j k Y l m
                    _ _ _ _ _ _ o p q r Z s t u v
                ]
                input_ids should be: [
                    a b c d e f X X X X X g h i j k Y Y Y l m
                    _ _ _ _ _ o p q r Z Z Z Z Z Z Z Z s t u v
                ]
                labels should be: [
                    a b c d e f _ _ _ _ _ g h i j k _ _ _ l m
                    _ _ _ _ _ o p q r _ _ _ _ _ _ _ _ s t u v
                ]
            Edge cases:
                * If tokens are same but audio token sizes are different, then cannot infer left or right padding
                ```python
                url1 = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-Audio/audio/glass-breaking-151256.mp3"
                audio1, _ = librosa.load(BytesIO(urlopen(url1).read()), sr=processor.feature_extractor.sampling_rate)
                url2 = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-Audio/audio/f2641_0_throatclearing.wav"
                audio2, _ = librosa.load(BytesIO(urlopen(url2).read()), sr=processor.feature_extractor.sampling_rate)
                prompts = [
                    "[INST] <|AUDIO|>\nWhat is that in this audio? [/INST]",
                    "[INST] <|AUDIO|>\nWhat is that in this audio? [/INST]",
                ]
                inputs = processor(text=prompts, audios=[audio1, audio2], return_tensors='pt', padding=True).to("cuda")
                    audio1 has 101 tokens, while audio2 has 72 tokens
                ```

                input_ids: [
                    a b c d X g h
                    i j Y k l m n
                ]
                where X is 3 tokens while Y is 5, this mean after merge
                if left-padding (batched generation)
                    input_ids should be: [
                        _ _ a b c d X X X g h
                        i j Y Y Y Y Y k l m n
                    ]
                elif (right padding) (training)
                    input_ids should be: [
                        a b c d X X X g h _ _
                        i j Y Y Y Y Y k l m n
                    ]
        """
        num_audios, max_audio_tokens, embed_dim = audio_features.shape
        audio_features_mask = torch.arange(max_audio_tokens).expand(
            num_audios, max_audio_tokens
        ).to(num_audio_tokens.device) < num_audio_tokens.unsqueeze(1)
        masked_audio_features = audio_features[audio_features_mask].view(-1, embed_dim)
        batch_size, sequence_length = input_ids.shape
        _left_padding = torch.any(attention_mask[:, 0] == 0)
        _right_padding = torch.any(attention_mask[:, -1] == 0)

        left_padding = True
        if batch_size > 1:
            if _left_padding and not _right_padding:
                left_padding = True
            elif not _left_padding and _right_padding:
                left_padding = False
            elif not _left_padding and not _right_padding:
                # both side is 1, so cannot tell
                left_padding = True
            else:
                # invalid attention_mask
                raise ValueError(
                    f"both side of attention_mask has zero, invalid. {attention_mask}"
                )

        # 1. Create a mask to know where special audio tokens are
        special_audio_token_mask = input_ids == self.tokenizer.default_speech_token
        num_special_audio_tokens = torch.sum(special_audio_token_mask, dim=-1)

        # In case the Audio model or the Language model has been offloaded to CPU, we need to manually
        # set the corresponding tensors into their correct target device.
        target_device = inputs_embeds.device
        attention_mask = attention_mask.to(target_device)
        input_ids = input_ids.to(target_device)
        num_audio_tokens = num_audio_tokens.to(target_device)
        batch_indices, non_audio_indices = torch.where(
            (input_ids != self.tokenizer.default_speech_token) & (attention_mask == 1)
        )

        # 2. Compute the positions where text should be written
        # Calculate new positions for text tokens in merged audio-text sequence.
        # `special_audio_token_mask` identifies audio tokens. Each audio token will be replaced by `audio_feat_lengths - 1` text tokens.
        # `torch.cumsum` computes how each audio token shifts subsequent text token positions.
        token_placeholder_num = torch.zeros_like(input_ids)
        token_placeholder_num[special_audio_token_mask] = num_audio_tokens.long() - 1
        token_placeholder_num = token_placeholder_num + 1
        new_token_positions = torch.cumsum(token_placeholder_num, -1) - 1
        max_token_num = token_placeholder_num.sum(-1).max()
        nb_audio_pad = max_token_num - 1 - new_token_positions[:, -1]
        if left_padding:
            new_token_positions += nb_audio_pad[:, None]  # offset for left padding
        text_to_overwrite = new_token_positions[batch_indices, non_audio_indices]
        batch_indices, non_audio_indices, text_to_overwrite = (
            batch_indices.to(target_device),
            non_audio_indices.to(target_device),
            text_to_overwrite.to(target_device),
        )

        # 3. Create the full embedding, already padded to the maximum position
        final_embedding = torch.zeros(
            batch_size,
            max_token_num,
            embed_dim,
            dtype=inputs_embeds.dtype,
            device=inputs_embeds.device,
        )
        final_attention_mask = torch.zeros(
            batch_size,
            max_token_num,
            dtype=attention_mask.dtype,
            device=inputs_embeds.device,
        )
        final_input_ids = torch.full(
            (batch_size, max_token_num),
            self.tokenizer.pad_token_id,
            dtype=input_ids.dtype,
            device=inputs_embeds.device,
        )

        # 4. Fill the embeddings based on the mask. If we have ["hey" "<audio>", "how", "are"]
        # we need to index copy on [0, 577, 578, 579] for the text and [1:576] for the audio features
        final_embedding[batch_indices, text_to_overwrite] = inputs_embeds[
            batch_indices, non_audio_indices
        ]
        final_attention_mask[batch_indices, text_to_overwrite] = attention_mask[
            batch_indices, non_audio_indices
        ]
        final_input_ids[batch_indices, text_to_overwrite] = input_ids[
            batch_indices, non_audio_indices
        ]
        final_labels = None
        if labels is not None:
            labels = labels.to(target_device)
            final_labels = torch.full(
                (batch_size, max_token_num),
                self.tokenizer.default_ignore_token,
                dtype=input_ids.dtype,
                device=inputs_embeds.device,
            ).to(torch.long)
            final_labels[batch_indices, text_to_overwrite] = labels[
                batch_indices, non_audio_indices
            ]
        # 5. Fill the embeddings corresponding to the audios. Anything that is still zeros needs filling
        audio_to_overwrite = torch.full(
            (batch_size, max_token_num),
            True,
            dtype=torch.bool,
            device=inputs_embeds.device,
        )
        audio_to_overwrite[batch_indices, text_to_overwrite] = False
        seq_indices = torch.arange(max_token_num).unsqueeze(0).to(target_device)
        seq_indices = seq_indices.expand(batch_size, max_token_num)

        if left_padding:
            # exclude padding on the left
            max_token_num = max_token_num.to(target_device)
            val = (max_token_num - seq_indices) <= (
                token_placeholder_num.sum(-1) - (attention_mask == 0).long().sum(-1)
            )[:, None]
        else:
            # exclude padding on the right
            val = (
                seq_indices
                < (
                    token_placeholder_num.sum(-1) - (attention_mask == 0).long().sum(-1)
                )[:, None]
            )

        audio_to_overwrite &= val

        if audio_to_overwrite.sum() != num_audio_tokens.sum():
            raise ValueError(
                f"The input provided to the model are wrong. The number of audio tokens is {num_special_audio_tokens} while"
                f" the number of audio given to the model is {num_audios}. This prevents correct indexing and breaks batch generation."
            )

        final_embedding[audio_to_overwrite] = (
            masked_audio_features.contiguous().reshape(-1, embed_dim).to(target_device)
        )
        final_attention_mask |= audio_to_overwrite
        position_ids = (final_attention_mask.cumsum(-1) - 1).masked_fill_(
            (final_attention_mask == 0), 1
        )

        return (
            final_embedding,
            final_attention_mask,
            final_labels,
            position_ids,
            final_input_ids,
        )
