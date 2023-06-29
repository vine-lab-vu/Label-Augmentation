import wandb


def initiate_wandb(args):
    if args.wandb:
        wandb.init(
            project=f"{args.wandb_project}", 
            entity=f"{args.wandb_entity}",
            name=f"{args.wandb_name}"
        )


def log_results(loss, dice, rmse_mean, best_rmse_mean, rmse_list, len_val_loader):
    wandb.log({
        'Train Loss': loss,
        'Dice Score': dice,
        'Mean RMSE': rmse_mean,
        'Best Mean RMSE': best_rmse_mean,
        'Label0': sum(rmse_list[0])/len_val_loader,
        'Label1': sum(rmse_list[1])/len_val_loader,
        'Label2': sum(rmse_list[2])/len_val_loader,
        'Label3': sum(rmse_list[3])/len_val_loader,
        'Label4': sum(rmse_list[4])/len_val_loader,
        'Label5': sum(rmse_list[5])/len_val_loader,
        'Label6': sum(rmse_list[6])/len_val_loader,
        'Label7': sum(rmse_list[7])/len_val_loader,
        'Label8': sum(rmse_list[8])/len_val_loader,
        'Label9': sum(rmse_list[9])/len_val_loader,
        'Label10': sum(rmse_list[10])/len_val_loader,
        'Label11': sum(rmse_list[11])/len_val_loader,
        'Label12': sum(rmse_list[12])/len_val_loader,
        'Label13': sum(rmse_list[13])/len_val_loader,
        'Label14': sum(rmse_list[14])/len_val_loader,
        'Label15': sum(rmse_list[15])/len_val_loader,
        'Label16': sum(rmse_list[16])/len_val_loader,
        'Label17': sum(rmse_list[17])/len_val_loader,
        'Label18': sum(rmse_list[18])/len_val_loader,
        'Label19': sum(rmse_list[19])/len_val_loader,
    })


def log_terminal(*a):
    file = open('log.txt','a')
    print(*a,file=file)