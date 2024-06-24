# Copyright 2024 HuggingFace Inc., THUDM, and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's transformers library and the THUDM's ChatGLM implementation.
# https://github.com/huggingface/transformers/blob/v4.40.0/examples/pytorch/summarization/run_summarization.py
# https://github.com/THUDM/ChatGLM-6B/blob/main/ptuning/main.py
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, Sequence, Tuple, Union

import numpy as np
from transformers.utils import is_jieba_available, is_nltk_available
from sklearn import metrics

from ...extras.constants import IGNORE_INDEX
from ...extras.packages import is_rouge_available


if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer


if is_jieba_available():
    import jieba  # type: ignore


if is_nltk_available():
    from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu


if is_rouge_available():
    from rouge_chinese import Rouge


@dataclass
class ComputeMetrics:
    r"""
    Wraps the tokenizer into metric functions, used in Seq2SeqPeftTrainer.
    """

    tokenizer: "PreTrainedTokenizer"

    def __call__(self, eval_preds: Sequence[Union[np.ndarray, Tuple[np.ndarray]]]) -> Dict[str, float]:
        r"""
        Uses the model predictions to compute metrics.
        """
        preds, labels = eval_preds
        score_dict = {"rouge-1": [], "rouge-2": [], "rouge-l": [], "bleu-4": []}

        preds = np.where(preds != IGNORE_INDEX, preds, self.tokenizer.pad_token_id)
        labels = np.where(labels != IGNORE_INDEX, labels, self.tokenizer.pad_token_id)

        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        for pred, label in zip(decoded_preds, decoded_labels):
            hypothesis = list(jieba.cut(pred))
            reference = list(jieba.cut(label))

            if len(" ".join(hypothesis).split()) == 0 or len(" ".join(reference).split()) == 0:
                result = {"rouge-1": {"f": 0.0}, "rouge-2": {"f": 0.0}, "rouge-l": {"f": 0.0}}
            else:
                rouge = Rouge()
                scores = rouge.get_scores(" ".join(hypothesis), " ".join(reference))
                result = scores[0]

            for k, v in result.items():
                score_dict[k].append(round(v["f"] * 100, 4))

            bleu_score = sentence_bleu([list(label)], list(pred), smoothing_function=SmoothingFunction().method3)
            score_dict["bleu-4"].append(round(bleu_score * 100, 4))

        return {k: float(np.mean(v)) for k, v in score_dict.items()}

@dataclass
class ComputeMetricsRiskRadar:
    r"""
    定制化的 gpt场景提示需要的指标计算
    """

    tokenizer: "PreTrainedTokenizer"

    def __call__(self, eval_preds: Sequence[Union[np.ndarray, Tuple[np.ndarray]]]) -> Dict[str, float]:
        r"""
        Uses the model predictions to compute metrics.
        """
        preds, labels = eval_preds
        score_dict = {"rouge-1": [], "rouge-2": [], "rouge-l": [], "整体_acc": []}

        preds = np.where(preds != IGNORE_INDEX, preds, self.tokenizer.pad_token_id)
        labels = np.where(labels != IGNORE_INDEX, labels, self.tokenizer.pad_token_id)

        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        # 去除边边角角
        decoded_labels = [ l.strip() for l in decoded_labels ]
        decoded_preds = [ l.strip() for l in decoded_preds ]

        for pred, label in zip(decoded_preds, decoded_labels):
            hypothesis = list(jieba.cut(pred))
            reference = list(jieba.cut(label))

            if len(" ".join(hypothesis).split()) == 0 or len(" ".join(reference).split()) == 0:
                result = {"rouge-1": {"f": 0.0}, "rouge-2": {"f": 0.0}, "rouge-l": {"f": 0.0}}
            else:
                rouge = Rouge()
                scores = rouge.get_scores(" ".join(hypothesis), " ".join(reference))
                result = scores[0]

            for k, v in result.items():
                score_dict[k].append(round(v["f"] * 100, 4))

            # bleu_score = sentence_bleu([list(label)], list(pred), smoothing_function=SmoothingFunction().method3)
            # score_dict["bleu-4"].append(round(bleu_score * 100, 4))
            
            # 新增acc部分
            acc = int(pred==label)
            score_dict["整体_acc"].append(round(acc * 100, 4))
        
        res = {k: float(np.mean(v)) for k, v in score_dict.items()}
            
        # 新增子类准确率计算
        unique_labels = np.unique(decoded_labels)
        precision, recall, f1_score, support = metrics.precision_recall_fscore_support(decoded_labels,
                                                                                       decoded_preds,
                                                                                       beta=1,
                                                                                       labels=unique_labels,
                                                                                       zero_division=0)
        f1_scores = {"整体_f1_score":[], "风险类_f1_score": []}  # 再加上各个类别的均值，比如风险类，经营勘察类
        for l,p,r,f,s, in zip(unique_labels, precision, recall, f1_score, support):
            f = round(f * 100, 4)
            
            short_l = "_".join([ i.split(":")[-1]  for i in l.split("\n")])
            short_l = short_l[:-1] if short_l[-1] == "_" else short_l
            key = short_l + "_f1_score"
            res[key] = f
            f1_scores["整体_f1_score"].append(f)
            
            # 新增风险类的计算
            second_class = short_l.strip('_')[-1]
            if second_class != '无':
                # 整体风险类
                f1_scores["风险类_f1_score"].append(f)
                
                # 各场景风险类
                scene = short_l.strip().split("_")[0] + "_风险类_f1_score"
                if scene not in f1_scores:
                    f1_scores[scene] = [f]
                else:
                    f1_scores[scene].append(f)
        
        res.update({k: float(np.mean(v)) for k,v in f1_scores.items()})
        return res