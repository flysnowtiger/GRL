
def adjust_lr(args, optimizer, epoch):
    lr = args.lr * (0.1 ** (epoch // args.lr_step))
    print(lr)
    for g in optimizer.param_groups:
        g['lr'] = lr * g.get('lr_mult', 1)