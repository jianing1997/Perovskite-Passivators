from argparse import ArgumentParser

def add_common_args(parser: ArgumentParser):
    opt_group = parser.add_argument_group("weighted retraining")
    opt_group.add_argument("--seed", type=int, required=True)
    opt_group.add_argument("--query_budget", type=int, required=True)
    opt_group.add_argument("--retraining_frequency", type=int, required=True)
    opt_group.add_argument(
        "--samples_per_model",
        type=int,
        default=1000,
        help="Number of samples to draw after each model retraining",
    )
    opt_group.add_argument("--gpu", action="store_true", help="Whether to use GPU")
    opt_group.add_argument("--result_root", type=str, required=True, help="root directory to store results in")
    opt_group.add_argument("--pretrained_model_file", type=str, required=True, help="path to pretrained model to use")
    opt_group.add_argument("--n_retrain_epochs", type=float, default=1.0)
    opt_group.add_argument("--n_init_retrain_epochs", type=float, default=None, help="None to use n_retrain_epochs, 0.0 to skip init retrain")
    opt_group.add_argument("--all_new", type=int, default=0)
    opt_group.add_argument("--invalid_score", type=float, default=-4.0)
    

    return parser
