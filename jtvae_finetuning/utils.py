""" Code for various 1-off functions """
import os
import sys
import uuid
import gzip
import pickle
import functools
import argparse
import datetime
import numpy as np
from scipy import stats
from tqdm.auto import tqdm
from pathlib import Path
import torch
import pytorch_lightning as pl




def is_pareto_efficient(costs, return_mask = True):
    """
    Find the pareto-efficient points (maximization based)
    :param costs: An (n_points, n_costs) array
    :param return_mask: True to return a mask
    :return: An array of indices of pareto-efficient points.
        If return_mask is True, this will be an (n_points, ) boolean array
        Otherwise it will be a (n_efficient_points, ) integer array of indices.
    """
    costs=-costs
    is_efficient = np.arange(costs.shape[0])
    n_points = costs.shape[0]
    next_point_index = 0  # Next index in the is_efficient array to search for
    while next_point_index<len(costs):
        nondominated_point_mask = np.any(costs<costs[next_point_index], axis=1)
        nondominated_point_mask[next_point_index] = True
        is_efficient = is_efficient[nondominated_point_mask]  # Remove dominated points
        costs = costs[nondominated_point_mask]
        next_point_index = np.sum(nondominated_point_mask[:next_point_index])+1
    if return_mask:
        is_efficient_mask = np.zeros(n_points, dtype = bool)
        is_efficient_mask[is_efficient] = True
        return is_efficient_mask
    else:
        return is_efficient


def PF_rank_generator(data):
  """Assign zero-based competition ranks to successive Pareto fronts.

  The first front receives rank 0. After assigning a front, the rank is
  incremented by that front's size, so the next front starts at the number of
  molecules contained in all better fronts.
  """
  base_data=data
  PF_rank=np.zeros((len(data),1))
  init_idx=np.arange(0,len(data))
  rank=0; # first pareto front will have rank=0
  while(len(data)!=0):
    PF_flag = is_pareto_efficient(costs=data, return_mask = True)
    M_idx = init_idx[PF_flag]
    init_idx=np.setdiff1d(init_idx,M_idx)
    data=base_data[init_idx]
    PF_rank[M_idx]=rank
    rank+=len(M_idx)

  return PF_rank.flatten()


# Weighting functions
class DataWeighter:
    weight_types = ["uniform", "rank", "dbas", "fb", "rwr", "cem-pi", "rank_min","rank_pf","rank_rwr","rank_pf_boost"]

    def __init__(self, hparams):

        if hparams.weight_type in ["uniform", "fb"]:
            self.weighting_function = DataWeighter.uniform_weights
        elif hparams.weight_type == "rank":
            self.weighting_function = functools.partial(
                DataWeighter.rank_weights, k_val=hparams.rank_weight_k
            )
        
        elif hparams.weight_type == "rank_min":
            self.weighting_function = functools.partial(
                DataWeighter.rank_weights_min, k_val=hparams.rank_weight_k
            )
        elif hparams.weight_type == "rank_pf":
            self.weighting_function = functools.partial(
                DataWeighter.rank_weights_pf, k_val=hparams.rank_weight_k
            )

        # Most other implementations are from:
        # https://github.com/dhbrookes/CbAS/blob/master/src/optimization_algs.py
        elif hparams.weight_type == "dbas":
            self.weighting_function = functools.partial(
                DataWeighter.dbas_weights,
                quantile=hparams.weight_quantile,
                noise=hparams.dbas_noise,
            )
        elif hparams.weight_type == "rwr":
            self.weighting_function = functools.partial(
                DataWeighter.rwr_weights, alpha=hparams.rwr_alpha
            )
        elif hparams.weight_type == "cem-pi":
            self.weighting_function = functools.partial(
                DataWeighter.cem_pi_weights, quantile=hparams.weight_quantile
            )
        elif hparams.weight_type == "rank_rwr":
            self.weighting_function = functools.partial(
                DataWeighter.rank_rwr_weights_pf, 
                k_val=hparams.rank_weight_k, 
                alpha=hparams.rwr_alpha
            )
        elif hparams.weight_type == "rank_pf_boost":
            self.weighting_function = functools.partial(
                DataWeighter.rank_weights_pf_boost, 
                k_val=hparams.rank_weight_k, 
                pce_threshold=hparams.pce_min_value, 
                boost_factor=hparams.pce_boost_factor
            )

        else:
            raise NotImplementedError

        self.weight_quantile = hparams.weight_quantile
        self.weight_type = hparams.weight_type

    @staticmethod
    def rank_rwr_weights_pf(properties: np.array, max_flag: np.array, k_val: float, alpha: float):
        """ 结合 Pareto 排名 (rank_pf) + RWR 计算权重 """
        
        # 1. 计算 Pareto 排名
        ranks = PF_rank_generator(properties * max_flag)
        rank_weights = 1.0 / (k_val * len(properties) + ranks)  # 计算 rank 权重
        
        # 2. 计算 RWR 权重
        pce_values = properties[:, 0]  # 假设 PCE 在第 1 列
        pce_max = np.max(pce_values)
        rwr_weights = np.exp(alpha * (pce_values - pce_max))  # RWR 计算公式
        
        # 3. 结合 Rank-PF 和 RWR 权重
        final_weights = rank_weights * rwr_weights  
        return final_weights / np.sum(final_weights)  # 归一化
    
    @staticmethod
    def rank_weights_pf_boost(properties: np.array, max_flag: np.array, k_val: float, pce_threshold: float, boost_factor: float):
        """ Pareto 排名 + 额外提升 PCE > threshold 的数据的权重 """
        
        # 1. 计算 Pareto 排名
        ranks = PF_rank_generator(properties * max_flag)
        rank_weights = 1.0 / (k_val * len(properties) + ranks)  # 计算 rank 权重
        
        # 2. 额外提升 PCE 高于 threshold 的数据权重
        boost_mask = properties[:, 0] >= pce_threshold  # 假设 PCE 在第 1 列
        rank_weights[boost_mask] *= boost_factor  # 仅对高 PCE 数据放大权重
        
        return rank_weights / np.sum(rank_weights)  # 归一化

    @staticmethod
    def normalize_weights(weights: np.array):
        """ Normalizes the given weights """
        return weights / np.mean(weights)

    @staticmethod
    def reduce_weight_variance(weights: np.array, data: np.array):
        """ Reduces the variance of the given weights via data replication """

        weights_new = []
        data_new = []
        for w, d in zip(weights, data):
            if w == 0.0:
                continue
            while w > 1:
                weights_new.append(1.0)
                data_new.append(d)
                w -= 1
            weights_new.append(w)
            data_new.append(d)

        return np.array(weights_new), np.array(data_new)

    @staticmethod
    def uniform_weights(properties: np.array):
        return np.ones_like(properties)

    @staticmethod
    def rank_weights(properties: np.array, k_val: float):
        """
        Calculates rank weights assuming maximization.
        Weights are not normalized.
        """
        if np.isinf(k_val):
            return np.ones_like(properties)
        ranks = np.argsort(np.argsort(-1 * properties))
        weights = 1.0 / (k_val * len(properties) + ranks)
        return weights
        
    @staticmethod
    def rank_weights_min(properties: np.array, k_val: float):
        """
        Calculates rank weights assuming minimization.
        Weights are not normalized.
        """
        if np.isinf(k_val):
            return np.ones_like(properties)
        ranks = np.argsort(np.argsort(properties))
        weights = 1.0 / (k_val * len(properties) + ranks)
        return weights
    
    @staticmethod
    def rank_weights_pf(properties: np.array,max_flag: np.array ,k_val: float):
        """
        Calculates rank weights assuming maximization.
        max_flag[0]=1 (-1) for maximization (minimization) of the properties1,
        """
        ranks = PF_rank_generator(properties*max_flag)
        weights = 1.0 / (k_val * len(properties) + ranks)
        return weights
    
    @staticmethod
    def dbas_weights(properties: np.array, quantile: float, noise: float):
        y_star = np.quantile(properties, quantile)
        if np.isclose(noise, 0):
            weights = (properties >= y_star).astype(float)
        else:
            weights = stats.norm.sf(y_star, loc=properties, scale=noise)
        return weights

    @staticmethod
    def cem_pi_weights(properties: np.array, quantile: float):

        # Find quantile cutoff
        cutoff = np.quantile(properties, quantile)
        weights = (properties >= cutoff).astype(float)
        return weights

    @staticmethod
    def rwr_weights(properties: np.array, alpha: float):

        # Subtract max property value for more stable calculation
        # It doesn't change the weights since they are normalized by the sum anyways
        prop_max = np.max(properties)
        weights = np.exp(alpha * (properties - prop_max))
        weights /= np.sum(weights)
        return weights

    @staticmethod
    def add_weight_args(parser: argparse.ArgumentParser):
        weight_group = parser.add_argument_group("weighting")
        weight_group.add_argument(
            "--weight_type",
            type=str,
            choices=DataWeighter.weight_types,
            default="uniform",
        )
        weight_group.add_argument(
            "--rank_weight_k",
            type=float,
            default=None,
            help="k parameter for rank weighting",
        )
        weight_group.add_argument(
            "--weight_quantile",
            type=float,
            default=None,
            help="quantile argument for dbas, cem-pi cutoffs",
        )
        weight_group.add_argument(
            "--dbas_noise",
            type=float,
            default=None,
            help="noise parameter for dbas (to induce non-binary weights)",
        )
        weight_group.add_argument(
            "--rwr_alpha", type=float, default=None, help="alpha value for rwr"
        )
        return parser


# Various pytorch functions
def _get_zero_grad_tensor(device):
    """ return a zero tensor that requires grad. """
    loss = torch.as_tensor(0.0, device=device)
    loss = loss.requires_grad_(True)
    return loss


def save_object(obj, filename):
    """ Function that saves an object to a file using pickle """

    result = pickle.dumps(obj)
    with gzip.GzipFile(filename, "wb") as dest:
        dest.write(result)
    dest.close()


def print_flush(text):
    print(text)
    sys.stdout.flush()


def update_hparams(hparams, model):

    # Make the hyperparameters match
    for k in model.hparams.keys():
        try:
            if vars(hparams)[k] != model.hparams[k]:
                print(
                    f"Overriding hparam {k} from {model.hparams[k]} to {vars(hparams)[k]}"
                )
                model.hparams[k] = vars(hparams)[k]
        except KeyError:  # not all keys match, it's ok
            pass

    # Add any new hyperparameters
    for k in vars(hparams).keys():
        if k not in model.hparams.keys():
            print(f'Adding missing hparam {k} with value "{vars(hparams)[k]}".')
            model.hparams[k] = vars(hparams)[k]


def add_default_trainer_args(parser, default_root=None):
    pl_trainer_grp = parser.add_argument_group("pl trainer")
    pl_trainer_grp.add_argument("--gpu", action="store_true")
    pl_trainer_grp.add_argument("--seed", type=int, default=0)
    pl_trainer_grp.add_argument("--root_dir", type=str, default=default_root)
    pl_trainer_grp.add_argument("--load_from_checkpoint", type=str, default=None)
    pl_trainer_grp.add_argument("--max_epochs", type=int, default=1000)


class SubmissivePlProgressbar(pl.callbacks.ProgressBar):
    """ progress bar with tqdm set to leave """

    def init_train_tqdm(self) -> tqdm:
        bar = tqdm(
            desc="Retraining Progress",
            initial=self.train_batch_idx,
            position=(2 * self.process_position),
            disable=self.is_disabled,
            leave=False,
            dynamic_ncols=True,
            file=sys.stdout,
            smoothing=0,
        )
        return bar
