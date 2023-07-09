import wandb


def initiate_wandb(args):
    if args.wandb:
        wandb.init(
            project=f"{args.wandb_project}", 
            entity=f"{args.wandb_entity}",
            name=f"{args.wandb_name}"
        )


def log_results(
    loss, loss_pixel, loss_geom, 
    dice, rmse_mean, best_rmse_mean, rmse_list, 
    train_loader_len, val_loader_len
    ):
    rmse_by_label = []
    for i in range(len(rmse_list)):
        sum, count = 0, 0
        for j in range(len(rmse_list[i])):
            if rmse_list[i][j] != 0:
               sum += rmse_list[i][j]
               count += 1
        rmse_by_label.append(sum/count)

    wandb.log({
        'Train Loss': loss/train_loader_len,
        'Pixel Loss': loss_pixel/train_loader_len,
        'Geometric Loss': loss_geom/train_loader_len,
        # 'Angle Loss':,
        'Dice Score': dice,
        'Mean RMSE': rmse_mean,
        'Best Mean RMSE': best_rmse_mean,
        'Label0': rmse_by_label[0],
        'Label1': rmse_by_label[1],
        'Label2': rmse_by_label[2],
        'Label3': rmse_by_label[3],
        'Label4': rmse_by_label[4],
        'Label5': rmse_by_label[5],
        'Label6': rmse_by_label[6],
        'Label7': rmse_by_label[7],
        'Label8': rmse_by_label[8],
        'Label9': rmse_by_label[9],
        'Label10': rmse_by_label[10],
        'Label11': rmse_by_label[11],
        'Label12': rmse_by_label[12],
        'Label13': rmse_by_label[13],
        'Label14': rmse_by_label[14],
        'Label15': rmse_by_label[15],
        'Label16': rmse_by_label[16],
        'Label17': rmse_by_label[17],
        'Label18': rmse_by_label[18],
        'Label19': rmse_by_label[19],
    })


def log_terminal(args, *a):
    file = open(f'{args.result_directory}/{args.wandb_name}/{args.wandb_name}_rmse_list.txt', 'a')
    print(*a, file=file)