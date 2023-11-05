def log_terminal(args, data_type, *a):
    file = open(f'{args.result_directory}/{args.wandb_name}/{args.wandb_name}_{data_type}.txt', 'a')
    print(*a, file=file)