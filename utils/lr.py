# -*- coding:utf-8 -*-
def poly_lr(epoch, max_epochs, initial_lr, exponent=0.9):
    return initial_lr * (1 - epoch / max_epochs)**exponent


def adjust_learning_rate(optimizer, epoch, initial_lr, max_epochs, exponent=0.9):
    lr = poly_lr(epoch, max_epochs, initial_lr, exponent)
    for i in range(len(optimizer.param_groups)):
        optimizer.param_groups[i]['lr'] = lr
    return lr

def step_learning_rate(optimizer, epoch, cur_lr, epochs, decay=0.1):
    if epoch in epochs:
        lr = cur_lr * decay
        for i in range(len(optimizer.param_groups)):
            optimizer.param_groups[i]['lr'] = lr
    else:
        lr = cur_lr
    return lr