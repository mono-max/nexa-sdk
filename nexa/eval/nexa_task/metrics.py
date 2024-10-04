import logging
import math
import random
import re
import string
import sacrebleu

from collections.abc import Iterable
from typing import List, Callable, Dict
import numpy as np


eval_logger = logging.getLogger("nexa-eval")


DEFAULT_METRIC = {
    "multiple_choice": ["acc", "acc_norm"],
    "generate_until": ["exact_match"],
}


### Metric ###


def mcc_fn(items):
    return items

def f1_fn(items):
    return items

def bleu_fn(items):
    return items

def ter_fn(items):
    return items

def brier_score_fn(items):
    return items

def acc_fn(items):
    return items

def acc_norm_fn(items):
    return items

def acc_mutual_info_fn(items):
    return items

def acc_all(items):
    # Only count as correct if all answers are labeled correctly for each question
    question_scoring_dict = {}
    preds = list(zip(*items))[0]
    docs = list(zip(*items))[1]

    for doc, pred in zip(docs, preds):
        paragraph_id = doc["idx"]["paragraph"]
        question_id = doc["idx"]["question"]
        if (paragraph_id, question_id) not in question_scoring_dict:
            question_scoring_dict[(paragraph_id, question_id)] = []

        gold_label = doc["label"] == 1

        question_scoring_dict[(paragraph_id, question_id)].append(gold_label == pred)
    acc = np.mean([int(all(x)) for x in question_scoring_dict.values()])
    return acc

def acc_all_stderr(items):
    # Only count as correct if all answers are labeled correctly for each question
    question_scoring_dict = {}
    preds = list(zip(*items))[0]
    docs = list(zip(*items))[1]

    for doc, pred in zip(docs, preds):
        question_id = doc["idx"]["question"]
        if question_id not in question_scoring_dict:
            question_scoring_dict[question_id] = []

        gold_label = doc["label"] == 1
        question_scoring_dict[question_id].append(gold_label == pred)

    acc = mean_stderr([int(all(x)) for x in question_scoring_dict.values()])
    return acc

def is_non_str_iterable(obj):
    return isinstance(obj, Iterable) and not isinstance(obj, str)

def _sacreformat(refs, preds):
    """Format refs and preds for sacrebleu corpus calculation. It is very particular"""
    # Sacrebleu expects (List[str], List[List[str])
    #   e.g. sacrebleu.corpus_bleu([pred_t], [[ref1_stream], [ref2_stream], ...])

    # Note [ref1_stream] is the first reference for each pred.
    # So lists are size N and (M, N) for N preds and M possible refs for each pred
    # This is a different order of dimensions that I would expect

    # We expect refs to be List[str] or List[List[str]], the outer list corresponding to preds
    # Must become List[List[str]] with the inner list corresponding to preds
    if not is_non_str_iterable(refs):
        refs = list(refs)
    if not is_non_str_iterable(refs[0]):
        refs = [[ref] for ref in refs]
    refs = list(zip(*refs))
    # Note the number of refs in each ref list much match the number of preds

    # We expect preds to be List[str] or List[List[str]]. Must become List[str]
    if not is_non_str_iterable(preds):
        preds = list(preds)
    if is_non_str_iterable(preds[0]):
        assert len(preds[0]) == 1, f"Pred must be a str, was {preds[0]}"
        preds = [pred[0] for pred in preds]

    return refs, preds

def bypass(items):
    return None

def exact_match_fn(**kwargs):
    return exact_match_hf_evaluate(**kwargs)

def mean_stderr(arr):
    return sample_stddev(arr) / math.sqrt(len(arr))

def sample_stddev(arr):
    mu = mean(arr)
    return math.sqrt(sum([(x - mu) ** 2 for x in arr]) / (len(arr) - 1))

def exact_match_hf_evaluate(
    predictions,
    references,
    regexes_to_ignore=None,
    ignore_case=False,
    ignore_punctuation=False,
    ignore_numbers=False,
):
    if regexes_to_ignore is not None:
        for s in regexes_to_ignore:
            predictions = np.array([re.sub(s, "", x) for x in predictions])
            references = np.array([re.sub(s, "", x) for x in references])
    else:
        predictions = np.asarray(predictions)
        references = np.asarray(references)

    if ignore_case:
        predictions = np.char.lower(predictions)
        references = np.char.lower(references)

    if ignore_punctuation:
        repl_table = string.punctuation.maketrans("", "", string.punctuation)
        predictions = np.char.translate(predictions, table=repl_table)
        references = np.char.translate(references, table=repl_table)

    if ignore_numbers:
        repl_table = string.digits.maketrans("", "", string.digits)
        predictions = np.char.translate(predictions, table=repl_table)
        references = np.char.translate(references, table=repl_table)

    score_list = predictions == references

    return {"exact_match": np.mean(score_list)}


metric_map = {
    "acc": acc_fn,
    "acc_norm": acc_norm_fn,
    "acc_all": acc_all,
    "acc_mutual_info": acc_mutual_info_fn,
    "bypass": bypass,
    "f1": f1_fn,
    "mcc": mcc_fn,
    "bleu": bleu_fn,
    "ter": ter_fn,
    "brier_score": brier_score_fn,
    "exact_match": exact_match_hf_evaluate,
}


### Aggregations ###


def bypass_agg(arr):
    return 999

def mean(arr):
    return sum(arr) / len(arr)

def f1_score(items):
    from sklearn.metrics import f1_score
    unzipped_list = list(zip(*items))
    golds = unzipped_list[0]
    preds = unzipped_list[1]
    fscore = f1_score(golds, preds)
    return np.max(fscore)

def bleu(items):
    refs = list(zip(*items))[0]
    preds = list(zip(*items))[1]
    refs, preds = _sacreformat(refs, preds)
    return sacrebleu.corpus_bleu(preds, refs).score

def ter(items):
    refs = list(zip(*items))[0]
    preds = list(zip(*items))[1]
    refs, preds = _sacreformat(refs, preds)
    return sacrebleu.corpus_ter(preds, refs).score

def brier_score(items):
    gold, predictions = list(zip(*items))
    bs, num_class = np.array(predictions).shape
    gold = list(gold)
    gold_one_hot = np.eye(num_class)[gold]
    return np.mean(np.sum((predictions - gold_one_hot) ** 2, axis=1))

def matthews_corrcoef(items):
    from sklearn.metrics import matthews_corrcoef
    unzipped_list = list(zip(*items))
    golds = unzipped_list[0]
    preds = unzipped_list[1]
    return matthews_corrcoef(golds, preds)


aggregation_map = {
    "mean": mean,
    "bypass": bypass_agg,
    "f1": f1_score,
    "matthews_corrcoef": matthews_corrcoef,
    "bleu": bleu,
    "ter": ter,
    "brier_score": brier_score,
}


metric_aggregation_map = {
    "acc": aggregation_map["mean"],
    "acc_norm": aggregation_map["mean"],
    "acc_all": aggregation_map["mean"],
    "acc_mutual_info": aggregation_map["mean"],
    "bypass": aggregation_map["bypass"],
    "f1": aggregation_map["f1"],
    "mcc": aggregation_map["matthews_corrcoef"],
    "bleu": aggregation_map["bleu"],
    "ter": aggregation_map["ter"],
    "brier_score": aggregation_map["brier_score"],
    "exact_match": aggregation_map["mean"],
}


# Functions to access the mappings


def get_metric(name: str) -> Callable:
    if name in metric_map:
        return metric_map[name]
    else:
        eval_logger.warning(f"Could not find metric '{name}' in metric_map!")
        raise KeyError(f"{name} not found in metric_map.")


def get_aggregation(name: str) -> Callable:
    if name in aggregation_map:
        return aggregation_map[name]
    else:
        eval_logger.warning(f"{name} not found in aggregation_map!")
        raise KeyError(f"{name} not found in aggregation_map.")


def get_metric_aggregation(name: str) -> Callable:
    if name in metric_aggregation_map:
        return metric_aggregation_map[name]
    else:
        eval_logger.warning(f"{name} metric is not assigned a default aggregation!")
        raise KeyError(f"{name} metric is not assigned a default aggregation!")


def is_higher_better(metric_name: str) -> bool:
    if metric_name in higher_is_better_map:
        return higher_is_better_map[metric_name]
    else:
        eval_logger.warning(f"higher_is_better not specified for metric '{metric_name}'!")
        raise KeyError(f"{metric_name} not found in higher_is_better_map.")


# stderr stuff


class _bootstrap_internal:
    def __init__(self, f, n) -> None:
        self.f = f
        self.n = n

    def __call__(self, v):
        i, xs = v
        rnd = random.Random()
        rnd.seed(i)
        res = []
        for _ in range(self.n):
            res.append(self.f(rnd.choices(xs, k=len(xs))))
        return res


def bootstrap_stderr(f, xs, iters):
    import multiprocessing as mp

    pool = mp.Pool(mp.cpu_count())
    # this gives a biased estimate of the stderr (i.e w/ the mean, it gives something
    # equivalent to stderr calculated without Bessel's correction in the stddev.
    # Unfortunately, I haven't been able to figure out what the right correction is
    # to make the bootstrap unbiased - i considered multiplying by sqrt(n/(n-1)) but
    # that would be ad-hoc and I can't prove that that would actually be an unbiased estimator)
    # Thankfully, shouldn't matter because our samples are pretty big usually anyways
    res = []
    chunk_size = min(1000, iters)
    from tqdm import tqdm

    print("bootstrapping for stddev:", f.__name__)
    for bootstrap in tqdm(
        pool.imap(
            _bootstrap_internal(f, chunk_size),
            [(i, xs) for i in range(iters // chunk_size)],
        ),
        total=iters // chunk_size,
    ):
        # sample w replacement
        res.extend(bootstrap)

    pool.close()
    return sample_stddev(res)


def stderr_for_metric(metric, bootstrap_iters: int):
    if bootstrap_iters <= 0:
        # return no function (don't compute stderr) if bootstrap iters = 0
        return None

    bootstrappable = [
        matthews_corrcoef,
        f1_score,
        bleu,
        ter,
    ]

    if metric in bootstrappable:
        return lambda x: bootstrap_stderr(metric, x, iters=bootstrap_iters)

    stderr = {mean: mean_stderr, acc_all: acc_all_stderr}

    return stderr.get(metric, None)


def pooled_sample_stderr(stderrs: List[float], sizes: List[int]):
    # Used to aggregate bootstrapped stderrs across subtasks in a group,
    # when we are weighting by the size of each subtask.
    #

    assert len(stderrs) == len(sizes)

    # formula source: https://en.wikipedia.org/wiki/Pooled_variance
    # and: https://stats.stackexchange.com/a/4841331
    # this empirically seems to match running `stderr_for_metric` on all instances
    # from the subtasks concatenated with each other.
    pooled_sample_var = (
        sum([(size - 1) * stderr**2 * size for size, stderr in zip(sizes, stderrs)])
    ) / (sum(sizes) - len(sizes))

    return np.sqrt(pooled_sample_var / sum(sizes))


def aggregate_subtask_metrics(metrics, sizes, weight_by_size=True):
    # A helper function that is used to aggregate
    # subtask scores cross-task.
    if not weight_by_size:
        sizes = [1] * len(sizes)

    assert len(metrics) == len(sizes)

    return sum([metric * size for metric, size in zip(metrics, sizes)]) / sum(sizes)