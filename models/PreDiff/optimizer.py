import torch


def build_optimizer(model, configs):
    optimizer_name = configs.get("optimizer", "adamw").lower()
    learning_rate = configs.get("learning_rate", 1e-4)
    weight_decay = configs.get("weight_decay", 1e-5)

    if optimizer_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=int(configs.get("scheduler_T_max", max(1, configs.get("epoch", 300)))),
    )
    return optimizer, scheduler
