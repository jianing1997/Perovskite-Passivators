import sys
import logging
import warnings
from tqdm.auto import tqdm, trange
import argparse
from pathlib import Path
import numpy as np
import torch
import pytorch_lightning as pl

# My imports
from jtvae_finetuning.chem.chem_data import (
    WeightedJTNNDataset,
)
from jtvae_finetuning.chem.chem_model import JTVAE
from jtvae_finetuning import utils
from jtvae_finetuning.chem.chem_utils import rdkit_quiet
from jtvae_finetuning.workflow import base as wr_base


logger = logging.getLogger("chem-opt")


def setup_logger(logfile):
    stream_handler = logging.StreamHandler(stream=sys.stdout)
    stream_handler.setLevel(logging.INFO)
    file_handler = logging.FileHandler(logfile)
    file_handler.setLevel(logging.DEBUG)
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    logger.setLevel(logging.DEBUG)


def retrain_model(model, datamodule, save_dir, version_str, num_epochs, gpu):

    # Make sure logs don't get in the way of progress bars
    pl._logger.setLevel(logging.CRITICAL)
    train_pbar = utils.SubmissivePlProgressbar(process_position=1)

    # Create custom saver and logger
    tb_logger = pl.loggers.TensorBoardLogger(
        save_dir=save_dir, version=version_str, name=""
    )
    checkpointer = pl.callbacks.ModelCheckpoint(save_last=True, monitor="loss/val",)

    # Handle fractional epochs
    if num_epochs < 1:
        max_epochs = 1
        limit_train_batches = num_epochs
    elif int(num_epochs) == num_epochs:
        max_epochs = int(num_epochs)
        limit_train_batches = 1.0
    else:
        raise ValueError(f"invalid num epochs {num_epochs}")

    # Create trainer
    trainer = pl.Trainer(
        gpus=1 if gpu else 0,
        max_epochs=max_epochs,
        limit_train_batches=limit_train_batches,
        limit_val_batches=1,
        checkpoint_callback=checkpointer,
        terminate_on_nan=True,
        logger=tb_logger,
        callbacks=[train_pbar],
        gradient_clip_val=20.0,  # Model is prone to large gradients
    )

    # Fit model
    trainer.fit(model, datamodule)


def _batch_decode_z_and_props(
    model: JTVAE,
    z: torch.Tensor,
    datamodule: WeightedJTNNDataset,
    args: argparse.Namespace,
    pbar: tqdm = None,
):
    """
    helper function to decode some latent vectors and calculate their properties
    """

    # Progress bar description
    if pbar is not None:
        old_desc = pbar.desc
        pbar.set_description("decoding")

    # Decode all points in a fixed decoding radius
    z_decode = []
    batch_size = 1
    for j in range(0, len(z), batch_size):
        with torch.no_grad():
            z_batch = z[j : j + batch_size]
            smiles_out = model.decode_deterministic(z_batch)
            if pbar is not None:
                pbar.update(z_batch.shape[0])
        z_decode += smiles_out

    # Now finding properties
    if pbar is not None:
        pbar.set_description("calc prop")

    # Calculate objective function values and choose which points to keep
    # Invalid points get a value of None
    if args.property2 is None:
        z_prop = [
            args.invalid_score if s is None else datamodule.train_dataset.prop_func(s)
            for s in z_decode
        ]
    else:
        z_prop = [
            [args.invalid_score, args.invalid_score] if s is None else [datamodule.train_dataset.prop_func(s),datamodule.train_dataset.prop_func2(s)]
            for s in z_decode
        ]

    # Now back to normal
    if pbar is not None:
        pbar.set_description(old_desc)

    return z_decode, z_prop


def latent_sampling_ranked(args, model, datamodule, num_queries_to_do, pbar=None):
    """Draw a fivefold latent pool and retain the highest-weight candidates."""

    # z_sample = torch.randn(num_queries_to_do, model.latent_dim, device=model.device)
    # z_decode, z_prop = _batch_decode_z_and_props(
    #     model, z_sample, datamodule, args, pbar=pbar
    # )

    z_sample = torch.randn(5*num_queries_to_do, model.latent_dim, device=model.device)
    z_decode, z_prop = _batch_decode_z_and_props(
        model, z_sample, datamodule, args, pbar=pbar
    )
    
    if args.weight_type=="rank_min":
        a=1
    else:
        a=-1
        
    print(np.shape(z_prop))
    print(np.shape(z_decode))
    # prop_val=np.array(prop_val)
    #tmp=np.argsort(a*z_prop)
    if args.weight_type=="rank_pf":
        tmp=np.argsort(-1*datamodule.data_weighter.weighting_function(z_prop,datamodule.flag))
    else:
        tmp=np.argsort([a*idx for idx in z_prop])
    # z_decode=z_decode[tmp,:]
    # z_prop=z_prop[tmp]
    z_decode=[z_decode[idx] for idx in tmp]
    z_prop=[z_prop[idx] for idx in tmp]

    return z_decode[0:num_queries_to_do], z_prop[0:num_queries_to_do]


def latent_sampling(args, model, datamodule, num_queries_to_do, pbar=None):
    """ Draws samples from latent space and appends to the dataset """

    z_sample = torch.randn(num_queries_to_do, model.latent_dim, device=model.device)
    z_decode, z_prop = _batch_decode_z_and_props(
        model, z_sample, datamodule, args, pbar=pbar
    )

    return z_decode, z_prop


def main_loop(args):
    
    print("Seeding")
    # Seeding
    pl.seed_everything(args.seed)
    
    print("Make results directory")
    # Make results directory
    result_dir = Path(args.result_root).resolve()
    result_dir.mkdir(parents=True)
    data_dir = result_dir / "data"
    data_dir.mkdir()
    setup_logger(result_dir / "log.txt")

    # Load data
    print("loading dataset")
    datamodule = WeightedJTNNDataset(args, utils.DataWeighter(args))
    datamodule.all_new=0
    if args.weight_type=="rank_pf":
        if args.property2=="SAS":
            max_flag = np.array([1,-1])
            
        else:
            max_flag = np.array([1,1])
        
        datamodule.flag=max_flag
        
    else:
        datamodule.flag=None
        
    datamodule.setup("fit")
    
    print("dataset loading completed")
    print(np.shape(datamodule.train_dataset.weights))
    print(np.shape(datamodule.train_dataset.data_properties))
    print(np.shape(datamodule.val_dataset.weights))
    print(np.shape(datamodule.val_dataset.data_properties))
    
    # Load model
    model = JTVAE.load_from_checkpoint(
        args.pretrained_model_file, vocab=datamodule.vocab
    )
    model.beta = model.hparams.beta_final  # Override any beta annealing

    # Set up some stuff for the progress bar
    num_retrain = int(np.ceil(args.query_budget / args.retraining_frequency))
    postfix = dict(
        retrain_left=num_retrain,
        best=-float("inf"),
        n_train=len(datamodule.train_dataset.data),
    )

    # Set up results tracking
    results = dict(
        opt_points=[],
        opt_point_properties=[],
        opt_model_version=[],
        params=str(sys.argv),
        sample_points=[],
        sample_versions=[],
        sample_properties=[],
    )

    # Main loop
    with tqdm(
        total=args.query_budget, dynamic_ncols=True, smoothing=0.0, file=sys.stdout
    ) as pbar:
        x_all_new=[]
        y_all_new=[]
        for ret_idx in range(num_retrain):
            pbar.set_postfix(postfix)
            pbar.set_description("retraining")

            # Decide whether to retrain
            samples_so_far = args.retraining_frequency * ret_idx

            # Optionally do retraining
            num_epochs = args.n_retrain_epochs
            if ret_idx == 0 and args.n_init_retrain_epochs is not None:
                num_epochs = args.n_init_retrain_epochs
            if num_epochs > 0:
                retrain_dir = result_dir / "retraining"
                version = f"retrain_{samples_so_far}"
                retrain_model(
                    model, datamodule, retrain_dir, version, num_epochs, args.gpu
                )
            del num_epochs

            # Update progress bar
            postfix["retrain_left"] -= 1
            pbar.set_postfix(postfix)

            # Draw samples for logs!
            if args.samples_per_model > 0:
                pbar.set_description("sampling")
                with trange(
                    args.samples_per_model, desc="sampling", leave=False
                ) as sample_pbar:
                    sample_x, sample_y = latent_sampling(
                        args, model, datamodule, args.samples_per_model,
                        pbar=sample_pbar
                    )
                
                # Append to results dict
                results["sample_points"].append(sample_x)
                results["sample_properties"].append(sample_y)
                results["sample_versions"].append(ret_idx)

            # Do querying!
            pbar.set_description("querying")
            num_queries_to_do = min(
                args.retraining_frequency, args.query_budget - samples_so_far
            )
            x_new, y_new = latent_sampling_ranked(
                args, model, datamodule, num_queries_to_do, pbar=pbar
            )

            # Update dataset
            if args.all_new==0:
                datamodule.append_train_data(x_new, y_new)
            else:
                del datamodule
                # gc.collect()
                
                datamodule = WeightedJTNNDataset(args, utils.DataWeighter(args))
                datamodule.all_new=0.1
                if args.weight_type=="rank_pf":
                    datamodule.flag = max_flag
                else:
                    datamodule.flag=None
                datamodule.setup("fit")
                x_all_new += x_new
                y_all_new += y_new
                datamodule.append_train_data(x_all_new, y_all_new)

            # Add new results
            results["opt_points"] += x_new
            results["opt_point_properties"] += y_new
            results["opt_model_version"] += [ret_idx]* len(x_new)

            # Save results
            np.savez_compressed(str(result_dir / "results.npz"), **results)

            # Keep a record of the dataset here
            new_data_file = (
                data_dir / f"train_data_iter{samples_so_far + num_queries_to_do}.txt"
            )
            with open(new_data_file, "w") as f:
                f.write("\n".join(datamodule.train_dataset.canonic_smiles))

            # postfix["best"] = max(postfix["best"], float(max(y_new)))
            postfix["n_train"] = len(datamodule.train_dataset.data)
            pbar.set_postfix(postfix)


if __name__ == "__main__":

    # Otherwise decoding fails completely
    rdkit_quiet()

    # arguments and argument checking
    parser = argparse.ArgumentParser()
    parser = WeightedJTNNDataset.add_model_specific_args(parser)
    parser = utils.DataWeighter.add_weight_args(parser)
    parser = wr_base.add_common_args(parser)

    # Pytorch lightning raises some annoying unhelpful warnings
    # in this script (because it is set up weirdly)
    # therefore we suppress warnings
    warnings.filterwarnings("ignore")

    # Parse args and run main code
    args = parser.parse_args()
    main_loop(args)
