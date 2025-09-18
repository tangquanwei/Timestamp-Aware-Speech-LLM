import torch
from torch.utils.data import IterableDataset
import kaldiio
from functools import partial
import torch.distributed as dist
import copy
import numpy as np
import os
import json
import random
import torchaudio.compliance.kaldi as kaldi
from loguru import logger
import math
import kaldi_native_fbank as knf
import torch_npu
from typing import Dict


class ASRFeatExtractor:
    def __init__(self, kaldi_cmvn_file):
        self.cmvn = CMVN(kaldi_cmvn_file) if kaldi_cmvn_file != "" else None
        self.fbank = KaldifeatFbank(
            num_mel_bins=80, frame_length=25, frame_shift=10, dither=0.0
        )

    def __call__(self, wav_path):
        sample_rate, wav_np = kaldiio.load_mat(wav_path)
        dur = wav_np.shape[0] / sample_rate
        fbank = self.fbank((sample_rate, wav_np))
        if self.cmvn is not None:
            fbank = self.cmvn(fbank)
        fbank = torch.from_numpy(fbank).float()
        return fbank, fbank.size(0)

    def pad_feat(self, xs, pad_value):
        # type: (List[Tensor], int) -> Tensor
        n_batch = len(xs)
        max_len = max([xs[i].size(0) for i in range(n_batch)])
        pad = (
            torch.ones(n_batch, max_len, *xs[0].size()[1:])
            .to(xs[0].device)
            .to(xs[0].dtype)
            .fill_(pad_value)
        )
        for i in range(n_batch):
            pad[i, : xs[i].size(0)] = xs[i]
        return pad


class CMVN:
    def __init__(self, kaldi_cmvn_file):
        self.dim, self.means, self.inverse_std_variences = self.read_kaldi_cmvn(
            kaldi_cmvn_file
        )

    def __call__(self, x, is_train=False):
        assert x.shape[-1] == self.dim, "CMVN dim mismatch"
        out = x - self.means
        out = out * self.inverse_std_variences
        return out

    def read_kaldi_cmvn(self, kaldi_cmvn_file):
        assert os.path.exists(kaldi_cmvn_file)
        stats = kaldiio.load_mat(kaldi_cmvn_file)
        assert stats.shape[0] == 2
        dim = stats.shape[-1] - 1
        count = stats[0, dim]
        assert count >= 1
        floor = 1e-20
        means = []
        inverse_std_variences = []
        for d in range(dim):
            mean = stats[0, d] / count
            means.append(mean.item())
            varience = (stats[1, d] / count) - mean * mean
            if varience < floor:
                varience = floor
            istd = 1.0 / math.sqrt(varience)
            inverse_std_variences.append(istd)
        return dim, np.array(means), np.array(inverse_std_variences)


class KaldifeatFbank:
    def __init__(self, num_mel_bins=80, frame_length=25, frame_shift=10, dither=1.0):
        self.dither = dither
        opts = knf.FbankOptions()
        opts.frame_opts.dither = dither
        opts.mel_opts.num_bins = num_mel_bins
        opts.frame_opts.snip_edges = True
        opts.mel_opts.debug_mel = False
        self.opts = opts

    def __call__(self, wav, is_train=False):
        if type(wav) is str:
            sample_rate, wav_np = kaldiio.load_mat(wav)
        elif type(wav) in [tuple, list] and len(wav) == 2:
            sample_rate, wav_np = wav
        assert len(wav_np.shape) == 1

        dither = self.dither if is_train else 0.0
        self.opts.frame_opts.dither = dither
        fbank = knf.OnlineFbank(self.opts)

        fbank.accept_waveform(sample_rate, wav_np.tolist())
        feat = []
        for i in range(fbank.num_frames_ready):
            feat.append(fbank.get_frame(i))
        if len(feat) == 0:
            print("Check data, len(feat) == 0", wav, flush=True)
            return np.zeros((0, self.opts.mel_opts.num_bins))
        feat = np.vstack(feat)
        return feat


class MultiTaskDataset(IterableDataset):
    def __init__(
        self, dataset_config, tokenizer=None, split="train", **kwargs,
    ):
        super().__init__()
        cmvn_path = dataset_config.cmvn_file
        self.feature_extractor = ASRFeatExtractor(cmvn_path)
        self.multitask_prompt_list = {}
        self.append_info_tasks = dataset_config.append_info_tasks
        with open(dataset_config.multitask_prompt_path) as f_prompt:
            for line in f_prompt:
                item = json.loads(line.strip())
                if item["task"] in self.multitask_prompt_list:
                    self.multitask_prompt_list[item["task"]].append(item["prompt"])
                else:
                    self.multitask_prompt_list[item["task"]] = [item["prompt"]]
        if split == "train":
            self.data_path = dataset_config.train_scp_file_path
        elif split == "val":
            self.data_path = dataset_config.dev_scp_file_path
        elif split == "test":
            self.data_path = dataset_config.test_scp_file_path
        else:
            raise ValueError("Split must be train val test")
        self.prompt_template = dataset_config.get("prompt_style", "{}")
        self.dataset_config = dataset_config
        self.tokenizer = tokenizer
        self.split = split
        self.max_audio_length = dataset_config.get("max_audio_length", 30)
        self.inference_mode = dataset_config.get("inference_mode", False)
        self.sample_rate = 16000
        self.mask_token_id=kwargs.get('mask_token_id',tokenizer.unk_token)
        self.mask_prob=kwargs.get('mask_prob',0.1)
        self.timestamp_num=kwargs.get('timestamp_num',30001)
        self.timestamp_ids = self._get_timestamp_ids(self.timestamp_num)


        # log Example data
        with open(os.path.join(self.data_path, "multitask.jsonl")) as f:
            for i in f:
                j = json.loads(i)
                logger.info(f"[Example] {j}")
                logger.info(f"[Prompt] {self.multitask_prompt_list[j['task']]}")
                break

    def __iter__(self):
        multitask_task_path = os.path.join(self.data_path, "multitask.jsonl")
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            num_workers = 1
            worker_id = 0
        else:
            num_workers = worker_info.num_workers
            worker_id = worker_info.id

        if dist.is_available() and dist.is_initialized():
            world_size = dist.get_world_size()
            rank = dist.get_rank()
        else:
            world_size = 1
            rank = 0

        total_num_workers = num_workers * world_size
        worker_rank = rank * num_workers + worker_id
        with open(multitask_task_path) as f_task:
            for data_index, line in enumerate(f_task):
                if (data_index % total_num_workers) == worker_rank:
                    try:
                        item = json.loads(line.strip())
                        ark_path = item["path"]
                        key = item["key"]
                        target = item["target"]
                        task = item["task"]
                        numpy_array = kaldiio.load_mat(ark_path)
                        audio_raw = numpy_array[1].astype(np.float32) / 32768
                        if (
                            len(audio_raw) / self.sample_rate > self.max_audio_length
                            or len(audio_raw) / self.sample_rate < 0.1
                        ):
                            continue
                        input_features, input_feature_length = self.feature_extractor(
                            ark_path
                        )
                        # print(input_features.shape, input_feature_length)
                        # exit(1)
                        prompt = random.choice(self.multitask_prompt_list[task])
                        prompt = self.prompt_template.format(prompt)
                        if task in self.append_info_tasks:
                            prompt = prompt.format(item[task])
                        prompt_ids = self.tokenizer.encode(prompt)
                        prompt_length = len(prompt_ids)
                        prompt_ids = torch.tensor(prompt_ids)

                        if not self.inference_mode:
                            target_ids = self.tokenizer.encode(target)
                            target_ids.append(self.tokenizer.eos_token_id)
                            target_ids = torch.tensor(target_ids)
                            input_ids = torch.cat([prompt_ids, target_ids])
                        else:
                            input_ids = prompt_ids
                        attention_mask = input_ids.ge(-1)
                        result = {
                            "input_ids": input_ids,
                            "attention_mask": attention_mask,
                            "input_features": input_features,
                            "input_feature_length": input_feature_length,
                            "key": key,
                            "target": target,
                        }

                        if not self.inference_mode:
                            labels = copy.deepcopy(input_ids)
                            labels[:prompt_length] = self.tokenizer.default_ignore_token
                            result["labels"] = labels
                        yield result
                    except Exception as e:
                        print(e)
                        print(data_index, item)
                        exit(1)

    def pad(self, sequence, max_length, padding_idx=0, padding_style="right"):
        if isinstance(sequence, (int, list, tuple)):
            if len(sequence) < max_length:
                if padding_style == "right":
                    sequence = sequence + [padding_idx] * (max_length - len(sequence))
                else:
                    sequence = [padding_idx] * (max_length - len(sequence)) + sequence
            else:
                sequence = sequence[:max_length]
        elif isinstance(sequence, torch.Tensor):
            if len(sequence) < max_length:
                if padding_style == "right":
                    sequence = torch.cat(
                        (
                            sequence,
                            torch.full(
                                (
                                    [max_length - len(sequence)]
                                    + list(sequence.size())[1:]
                                ),
                                padding_idx,
                            ),
                        )
                    )
                else:
                    sequence = torch.cat(
                        (
                            torch.full(
                                (
                                    [max_length - len(sequence)]
                                    + list(sequence.size())[1:]
                                ),
                                padding_idx,
                            ),
                            sequence,
                        )
                    )
            else:
                sequence = sequence[:max_length]
        elif isinstance(sequence, np.ndarray):
            if len(sequence) < max_length:
                if padding_style == "right":
                    sequence = np.concatenate(
                        (
                            sequence,
                            np.full(
                                (max_length - len(sequence),) + sequence.shape[1:],
                                padding_idx,
                            ),
                        )
                    )
                else:
                    sequence = np.concatenate(
                        (
                            np.full(
                                (max_length - len(sequence),) + sequence.shape[1:],
                                padding_idx,
                            ),
                            sequence,
                        )
                    )
            else:
                sequence = sequence[:max_length]
        else:
            raise Exception("Type mismatch during padding!")
        return sequence

    def extract_fbank(self, waveform):
        fbank_features = kaldi.fbank(
            waveform,
            num_mel_bins=self.dataset_config["fbankConfig"]["num_mel_bins"],
            frame_length=self.dataset_config["fbankConfig"]["frame_length"],
            frame_shift=self.dataset_config["fbankConfig"]["frame_shift"],
            dither=self.dataset_config["fbankConfig"]["dither"]
            if self.split == "train"
            else 0,
            window_type=self.dataset_config["fbankConfig"]["window_type"],
            use_energy=self.dataset_config["fbankConfig"]["use_energy"],
            low_freq=self.dataset_config["fbankConfig"]["low_freq"],
            high_freq=self.dataset_config["fbankConfig"]["high_freq"],
            htk_compat=self.dataset_config["fbankConfig"]["htk_compat"],
        )
        return fbank_features

    def collator(self, samples):
        assert samples is not None
        if self.mask and not self.inference_mode:
            samples = self._apply_timestamp_mask(
                samples
            )
        if self.inference_mode:
            padding_style = "left"
        else:
            padding_style = "right"
        padding_style = "left"
        input_feature_length = torch.stack(
            [torch.tensor(s["input_feature_length"]) for s in samples]
        )
        input_ids_max_length = max([s["input_ids"].shape[0] for s in samples])
        input_ids = torch.stack(
            [
                self.pad(
                    s["input_ids"],
                    input_ids_max_length,
                    self.tokenizer.pad_token_id,
                    padding_style=padding_style,
                )
                for s in samples
            ]
        )
        attention_mask = torch.stack(
            [
                self.pad(
                    s["attention_mask"],
                    input_ids_max_length,
                    False,
                    padding_style=padding_style,
                )
                for s in samples
            ]
        )
        input_features_max_length = max([s["input_features"].shape[0] for s in samples])
        input_features = torch.stack(
            [
                self.pad(s["input_features"], input_features_max_length, 0.0)
                for s in samples
            ]
        )
        result = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "input_features": input_features,
            "input_feature_length": input_feature_length,
        }

        if self.inference_mode:
            result["keys"] = [s["key"] for s in samples]
            result["targets"] = [s["target"] for s in samples]
        else:
            result["labels"] = torch.stack(
                [
                    self.pad(
                        s["labels"],
                        input_ids_max_length,
                        self.tokenizer.default_ignore_token,
                        padding_style=padding_style,
                    )
                    for s in samples
                ]
            )
        return result

    def _get_timestamp_ids(self, timestamp_num) -> set:
        timestamp_ids = set()
        for i in range(timestamp_num):  # 0.00 to 5.00
            token_id = 151646 + i
            if token_id != self.tokenizer.unk_token:
                timestamp_ids.add(token_id)
        return timestamp_ids

    def _is_timestamp(self, token_id: int) -> bool:
        return token_id in self.timestamp_ids

    def _apply_timestamp_mask(
        self,
        sample: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        mask_token_id,mask_prob= self.mask_token_id, self.mask_prob
        input_ids = sample["input_ids"].clone()
        labels = input_ids.clone()
        is_ts = torch.tensor(
            [self._is_timestamp(tid.item()) for tid in input_ids], dtype=torch.bool
        )
        probability_matrix = torch.full_like(input_ids, 0.0, dtype=torch.float)
        probability_matrix[is_ts] = mask_prob

        masked_indices = torch.bernoulli(probability_matrix).bool()

        input_ids[masked_indices] = mask_token_id

        sample["input_ids"] = input_ids
        sample["labels"] = labels
        return sample


class MultiTaskDynamicBatchDataset(IterableDataset):
    def __init__(self, dataset: IterableDataset, window_class) -> None:
        super().__init__()
        self.dp = dataset

        assert window_class is not None
        self.window_class = window_class
        self.collator = self.dp.collator
        self._buffer = []

    def __iter__(self):
        for elem in self.dp:
            if not self.window_class(elem, self._buffer):
                self._buffer.append(elem)
            else:
                if len(self._buffer) > 0:
                    yield self._buffer
                del self._buffer
                self._buffer = [elem]
        if len(self._buffer) > 0:
            yield self._buffer
        del self._buffer
        self._buffer = []


def window_class(elem, buffer, max_frame_length, ds_rate):
    if len(buffer) == 0:
        return True
    max_frame = max(
        len(elem["input_ids"]) + (elem["input_feature_length"] // ds_rate) - 1,
        max(
            [
                len(_["input_ids"]) + (_["input_feature_length"] // ds_rate) - 1
                for _ in buffer
            ]
        ),
    )
    return (len(buffer) + 1) * max_frame > max_frame_length


def get_speech_dataset(dataset_config, tokenizer, split):
    dataset = MultiTaskDataset(dataset_config, tokenizer, split)
    if split == "train":
        dataset = MultiTaskDynamicBatchDataset(
            dataset,
            partial(
                window_class,
                max_frame_length=dataset_config.train_max_frame_length,
                ds_rate=dataset_config.ds_rate,
            ),
        )
    else:
        dataset = MultiTaskDynamicBatchDataset(
            dataset,
            partial(
                window_class,
                max_frame_length=dataset_config.eval_max_frame_length,
                ds_rate=dataset_config.ds_rate,
            ),
        )
    return dataset
