import wandb


def initiate_wandb(args):
    if args.wandb:
        wandb.init(
            project=f"{args.wandb_project}", 
            entity=f"{args.wandb_entity}",
            name=f"{args.wandb_name}"
        )


def log_results(
    loss, dice, rmse_mean, best_rmse_mean, rmse_mean_by_label,
    train_loader_len, val_loader_len
    ):
    wandb.log({
        'Train Loss': loss/train_loader_len,
        'Dice Score': dice,
        'Mean RMSE': rmse_mean,
        'Best Mean RMSE': best_rmse_mean,
        'Label0': rmse_mean_by_label[0],
        'Label1': rmse_mean_by_label[1],
        'Label2': rmse_mean_by_label[2],
        'Label3': rmse_mean_by_label[3],
        'Label4': rmse_mean_by_label[4],
        'Label5': rmse_mean_by_label[5],
        'Label6': rmse_mean_by_label[6],
        'Label7': rmse_mean_by_label[7],
        'Label8': rmse_mean_by_label[8],
        'Label9': rmse_mean_by_label[9],
        'Label10': rmse_mean_by_label[10],
        'Label11': rmse_mean_by_label[11],
        'Label12': rmse_mean_by_label[12],
        'Label13': rmse_mean_by_label[13],
        'Label14': rmse_mean_by_label[14],
        'Label15': rmse_mean_by_label[15],
        'Label16': rmse_mean_by_label[16],
        'Label17': rmse_mean_by_label[17],
        'Label18': rmse_mean_by_label[18],
        'Label19': rmse_mean_by_label[19],
    })


def log_results_with_angle(
    loss, dice, rmse_mean, best_rmse_mean, rmse_mean_by_label, best_angle_diff, angle_value,
    train_loader_len, val_loader_len, angle_len
    ):
    wandb.log({
        'Train Loss': loss/train_loader_len,
        'Dice Score': dice,
        'Mean RMSE': rmse_mean,
        'Best Mean RMSE': best_rmse_mean,
        'Mean Angle Difference': angle_value[3]/(val_loader_len*angle_len),
        'Best Mean Angle Difference': best_angle_diff/(val_loader_len*angle_len),
        'dFA': angle_value[0]/(val_loader_len),
        'pTA': angle_value[1]/(val_loader_len),
        'FTA': angle_value[2]/(val_loader_len),
        'Label0': rmse_mean_by_label[0],
        'Label1': rmse_mean_by_label[1],
        'Label2': rmse_mean_by_label[2],
        'Label3': rmse_mean_by_label[3],
        'Label4': rmse_mean_by_label[4],
        'Label5': rmse_mean_by_label[5],
        'Label6': rmse_mean_by_label[6],
        'Label7': rmse_mean_by_label[7],
        'Label8': rmse_mean_by_label[8],
        'Label9': rmse_mean_by_label[9],
        'Label10': rmse_mean_by_label[10],
        'Label11': rmse_mean_by_label[11],
        'Label12': rmse_mean_by_label[12],
        'Label13': rmse_mean_by_label[13],
        'Label14': rmse_mean_by_label[14],
        'Label15': rmse_mean_by_label[15],
        'Label16': rmse_mean_by_label[16],
        'Label17': rmse_mean_by_label[17],
        'Label18': rmse_mean_by_label[18],
        'Label19': rmse_mean_by_label[19],
    })


def log_terminal(args, *a):
    file = open(f'{args.result_directory}/{args.wandb_name}/{args.wandb_name}_rmse_list.txt', 'a')
    print(*a, file=file)