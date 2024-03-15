"""
client_num = 100

epoch = 100

val_freq = 5
save_freq = 5

poison_ratio =  0.05
lamda = 2


"""
from Node import Global_node, Local_node2, init_model, train_merge, train_clean, validate, init_prompter
from Args import parse_option
from Model import vit, ResNet50

from Utils import get_map_indices
import Data
from copy import deepcopy
from torch.utils.data import DataLoader
import numpy as np



def init_original_data(args):
    dataset_name = args.dataset
    train_dataset, test_dataset, class_names, num_classes = Data.get_full_data(
        dataset_name)

    client_num = args.client_num

    if args.mode == 'iid':
        subset_idx_list = Data.divide_data_iid(len(train_dataset), client_num)
        pass
    elif args.mode == 'noiid':
        # subset_idx_list = Data.divide_data_noniid(train_dataset.targets,client_num,5)
        subset_idx_list = Data.divide_data_dirichlet(
            train_dataset.targets, num_classes, client_num, args.alpha)
        pass
    return train_dataset, test_dataset, class_names, num_classes, subset_idx_list


def init_node_data(node_id, train_dataset, test_dataset, subset_idx_list, args):
    idx_list = subset_idx_list[node_id]
    temp_data, temp_targets = [train_dataset.data[idx] for idx in idx_list], [
        train_dataset.targets[idx] for idx in idx_list]
    local_train_dataset = Data.CustomDataset(
        deepcopy(temp_data), deepcopy(temp_targets))
    del temp_data, temp_targets

    poison_ratio = args.poison_ratio
    trigger_size = args.trigger_size
    trigger_pos = args.trigger_pos
    target_class = args.target_class
    batch_size = args.batch_size
    num_workers = args.num_workers
    # print(trigger_pos)
    dataset_name = args.dataset

    train_clean_dataset = local_train_dataset
    train_merge_dataset = Data.get_train_merge_dataset(
        train_clean_dataset, trigger_pos=trigger_pos,
        trigger_size=trigger_size, target_classes=target_class,
        poison_ratio=poison_ratio, dataset_name=dataset_name)

    test_clean_dataset = test_dataset
    test_backdoor_dataset = Data.get_test_backdoor_dataset(
        test_clean_dataset, trigger_pos=trigger_pos,
        trigger_size=trigger_size, target_classes=target_class)

    train_merge_loader = DataLoader(
        train_merge_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True, shuffle=True)
    train_clean_loader = DataLoader(
        train_clean_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True, shuffle=False)
    test_clean_loader = DataLoader(test_clean_dataset, batch_size=batch_size,
                                   num_workers=num_workers, pin_memory=True, shuffle=False)
    test_backdoor_loader = DataLoader(
        test_backdoor_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True, shuffle=False)

    return train_merge_loader, train_clean_loader, test_clean_loader, test_backdoor_loader


def init_model(args):

    # model = vit()
    model = ResNet50()
    model.to(args.device)
    
    return model


def main(args):
    np.random.seed(42)
    data_save = {}
    device = args.device
    dataset_name = args.dataset

    client_num = args.client_num
    round = args.round
    epochs = args.epochs
    select_num = args.select_num
    poison_client_num = args.poison_client_num
    
    client_num = 100
    poison_client_num = 20
    
    epochs = 100
    val_freq = 5
    save_freq = 5
    

    train_dataset, test_dataset, class_names, num_classes, subset_idx_list = init_original_data(
        args)
    poison_client_idx = np.random.choice(
        range(0, client_num), poison_client_num, replace=False)
    clean_client_idx = [i for i in range(
        client_num) if i not in poison_client_idx]
    print(len(poison_client_idx), len(clean_client_idx))
    
    node_list = []

    for i in range(client_num):
        data_save['node_{}_acc'.format(i)] = []
        data_save['node_{}_asr'.format(i)] = []
        data_save['node_{}_loss'.format(i)] = []
        total_steps = int(
            len(subset_idx_list[i]) * 100 * 1  / 64)
        temp_node = Local_node2(i, args, total_steps)
        node_list.append(temp_node)
        pass
    
    # 初始化将要使用的大模型
    model = init_model(args)
    test_clean_loader = Data.DataLoader(
        test_dataset, args.batch_size, num_workers=16, shuffle=True)
    indices = get_map_indices(model, test_clean_loader, num_classes, device)

    for i in range(1):

        select_idx = np.random.choice(
            range(0, client_num), select_num, replace=False)
        select_idx = [ii for ii in range(client_num)]
        print('Round {}/{} the selected nodes is '.format(i+1, round), select_idx)
        select_idx_list = [] # 记录选的是哪几个客户端，因为客户端权重跟其样本数量有关
        for node_id in select_idx:
            print('node_id {}'.format(node_id))
            # 准备数据
            is_poison = False if node_id in clean_client_idx else True
            now_node = node_list[node_id]
            train_merge_loader, train_clean_loader, test_clean_loader, test_backdoor_loader = \
                init_node_data(node_id, train_dataset,
                               test_dataset, subset_idx_list, args)
            # 加载模型
            # 已经加载完毕
            model.to(device)
            # continue
            # 开始训练
            for now_epoch in range(epochs):
                # print(now_epoch,end=' ')
                prev_prompt = None
                global_prompter_current = None
                if  is_poison:
                    loss, top1 = train_merge(indices, train_merge_loader, model, prev_prompt, global_prompter_current, now_node.prompter, now_node.optimizer,
                                             now_node.scheduler, now_node.criterion, now_node.epoch + 1, now_node.args)
                else:
                    loss, top1 = train_clean(indices, train_clean_loader, model, now_node.prompter, now_node.optimizer,
                                             now_node.scheduler, now_node.criterion, now_node.epoch + 1, now_node.args)
                now_node.epoch += 1
                # print(top1,losses)
                # acc = 0
                data_save['node_{}_loss'.format(i)].append(loss)
                

                if (now_epoch +1) % val_freq == 0:
                    acc = validate(indices, test_clean_loader, model,
                                now_node.prompter, now_node.criterion, now_node.args)
                    asr = validate(indices, test_backdoor_loader, model,
                                now_node.prompter, now_node.criterion, now_node.args)

                    if acc > now_node.best_acc:
                        now_node.save_checkpoint(isbest=True)
                        now_node.best_acc = acc
                        now_node.bese_asr = asr
                    
                    data_save['node_{}_acc'.format(i)].append(acc)
                    data_save['node_{}_asr'.format(i)].append(asr)

                
                    if is_poison :
                        desc = 'Round {:>3d}/{:3d} Node_{:<3d} Poison Epoch {:3d}  Loss is {:4.5f} Acc is {:3.2f} Asr is {:3.2f}'
                    else :
                        desc = 'Round {:>3d}/{:3d} Node_{:<3d} Clean  Epoch {:3d}  Loss is {:4.5f} Acc is {:3.2f} Asr is {:3.2f}'
                    print(desc.format(
                        i+1, round, node_id, now_node.epoch,loss, acc, asr))
                else :
                    if is_poison :
                        desc = 'Round {:>3d}/{:3d} Node_{:<3d} Poison Epoch {:3d}  Loss is {:4.5f}'
                    else :
                        desc = 'Round {:>3d}/{:3d} Node_{:<3d} Clean  Epoch {:3d}  Loss is {:4.5f}'
                    print(desc.format(
                    i+1, round, node_id, now_node.epoch,  loss))

                if now_epoch+1 % save_freq == 0:
                    now_node.save_checkpoint()



            # 保存模型
            pass
            # node_list[node_id] = now_node # 可要可不要吧？1                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               
            select_idx_list.append(node_id)
        # 聚合并且测试
    

if __name__ == '__main__':
    fuck_args = parse_option()
    main(fuck_args)
    pass
