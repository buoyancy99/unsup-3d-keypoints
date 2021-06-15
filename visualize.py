from experiments import algo_registry, get_args


if __name__ == '__main__':
    args = get_args()
    experiment = algo_registry[args.algo](args)
    experiment.visualize()
