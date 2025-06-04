import argparse
import os
import pickle
from collections import defaultdict
from os.path import exists
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import wandb
from torch.utils.data import TensorDataset, DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.utils import resample
from tqdm import tqdm
from sklearn import metrics
# from transformers import set_seed
import random
from probing_utils import extract_internal_reps_specific_layer_and_token, compile_probing_indices, \
    load_model_and_validate_gpu, get_probing_layer_names, LIST_OF_DATASETS, LIST_OF_MODELS,test_classifier_lstm, \
    MODEL_FRIENDLY_NAMES, LIST_OF_PROBING_LOCATIONS, compute_metrics_probing, prepare_for_probing,LIST_OF_MODELS_PATH


def parse_args_and_init_wandb():
    parser = argparse.ArgumentParser(
        description='Probe for hallucinations and create plots')
    parser.add_argument("--model", choices=LIST_OF_MODELS)
    parser.add_argument("--probe_at",
                        choices=LIST_OF_PROBING_LOCATIONS)
    # important args for the results of the papaer
    parser.add_argument("--seeds", nargs='+', type=int)
    parser.add_argument("--n_samples", help="size of validation data", default='all')
    parser.add_argument("--layer", type=int)
    parser.add_argument("--token", type=str)
    parser.add_argument("--save_clf", action='store_true', default=False, help="Whether to save the clf. If true, will look for a classifier before training and load it if exists.")
    parser.add_argument("--only_train", action='store_true', default=False,
                        help="Whether to save the clf. If true, will look for a classifier before training and load it if exists.")

    parser.add_argument("--dataset", choices=LIST_OF_DATASETS, required=True)
    parser.add_argument("--test_dataset", choices=LIST_OF_DATASETS, required=False, default=None)

    args = parser.parse_args()
    os.environ["WANDB_MODE"] = "offline"
    if args.test_dataset is None:
        wandb.init(
            project="probe_hallucinations_specific",
            config=vars(args)
        )
    else:
        wandb.init(
            project="probe_hallucinations_generalization",
            config=vars(args)
        )

    return args
import torch
import torch.nn as nn

class UniLSTMAttention(nn.Module):
    def __init__(self, input_dim=4096, lstm_hidden=256):
        super().__init__()
        # 输入降维层（4096 -> 512）
        self.embed = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.LayerNorm(512)
        )

        # 单向LSTM
        self.lstm = nn.LSTM(
            input_size=512,
            hidden_size=lstm_hidden,
            num_layers=1,  # 单层简化
            batch_first=True,
            dropout=0.2
        )

        # 注意力机制
        self.attention = nn.Sequential(
            nn.Linear(lstm_hidden, 64),  # 输入维度减半（单向）
            nn.Tanh(),
            nn.Linear(64, 1),
            nn.Softmax(dim=1)
        )

        # 分类头
        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden, 1)  # 输入维度减半
        )

    def forward(self, x):
        # 1. 降维
        x = self.embed(x)  # [batch, 32, 512]

        # 2. 单向LSTM
        lstm_out, _ = self.lstm(x)  # [batch, 32, lstm_hidden]

        # 3. 注意力
        attn_weights = self.attention(lstm_out)  # [batch, 32, 1]
        context = torch.sum(lstm_out * attn_weights, dim=1)  # [batch, lstm_hidden]

        # 4. 输出
        return self.fc(context)
    def predict(self, x):
        # 1. 降维
        x = self.embed(x)  # [batch, 32, 512]

        # 2. 单向LSTM
        lstm_out, _ = self.lstm(x)  # [batch, 32, lstm_hidden]

        # 3. 注意力
        attn_weights = self.attention(lstm_out)  # [batch, 32, 1]
        context = torch.sum(lstm_out * attn_weights, dim=1)  # [batch, lstm_hidden]

        # 4. 输出
        preds = (torch.sigmoid(self.fc(context)) > 0.5).float()
        return preds

class LSTMBinaryClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super().__init__()
        self.fc_1 = nn.Linear(input_size,hidden_size)
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0  # 仅在多层时启用Dropout
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x: [batch_size, 32, input_dim]
        x = self.fc_1(x)
        _, (h_n, _) = self.lstm(x)  # h_n: [num_layers, batch_size, hidden_dim]
        last_hidden = h_n[-1]  # 取最后一层的隐藏状态
        output = self.fc(last_hidden)  # 输出形状: [batch_size, 1]
        return output
    def predict(self, x):
        x = self.fc_1(x)
        _, (h_n, _) = self.lstm(x)  # h_n: [num_layers, batch_size, hidden_dim]
        last_hidden = h_n[-1]  # 取最后一层的隐藏状态
        output = self.fc(last_hidden)
        preds = (torch.sigmoid(output) > 0.5).float()
        return preds


def train_epoch(model, dataloader, criterion, optimizer):
    model.train()
    total_loss = 0
    print("Training...")

    for x_batch, y_batch in tqdm(dataloader):
        x_batch, y_batch = x_batch.to("cuda"), y_batch.to("cuda")
        optimizer.zero_grad()
        outputs = model(x_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # 梯度裁剪
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def evaluate(model, dataloader, criterion):
    model.eval()
    total_loss, correct = 0, 0
    with torch.no_grad():
        print("evaluating...")
        auc_eval = []
        for x_batch, y_batch in tqdm(dataloader):
            x_batch, y_batch = x_batch.to("cuda"), y_batch.to("cuda")
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            total_loss += loss.item()
            preds = (torch.sigmoid(outputs) > 0.5).float()
            correct += (preds == y_batch).sum().item()
            fpr_for_auc, tpr_for_auc, thresholds = metrics.roc_curve(y_batch.cpu().detach(),
                                                                     torch.sigmoid(outputs.cpu().detach()))
            auc = metrics.auc(fpr_for_auc, tpr_for_auc)
            auc_eval.append(auc)

    return total_loss / len(dataloader), correct / len(dataloader.dataset),np.array(auc_eval).mean()
# 训练循环

def init_and_train_classifier_lstm(if_save,X_train, y_train,path,X_test_, y_test_,start_end=[]):
    data={}
    train_losses = []
    val_losses = []
    epochs = []
    # train_accuracies = []  # 如果需要训练集准确率
    val_accuracies = []
    num_epochs = 30
    batch_size = 128
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    y_train=torch.FloatTensor(y_train.reshape(-1, 1))
    X_train=X_train[:,start_end[0]:start_end[1]].float()
    model = LSTMAttentionClassifier().to(device)
    #model = LSTMBinaryClassifier(input_size=4096, hidden_size=256, num_layers=1).to(device)
    criterion = nn.BCEWithLogitsLoss()  # 内置Sigmoid，适用于二元分类
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)  # L2正则化
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2)
    dataset = TensorDataset(X_train, y_train)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    best_model_weights = None
    best_val_loss = np.inf
    patience_counter = 0
    patience = 5
    min_delta=0.001
    for epoch in tqdm(range(num_epochs)):
        train_loss = train_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc,val_auc = evaluate(model, val_loader, criterion)
        # test_classifier_lstm(model, X_test_, y_test_)
        scheduler.step(val_loss)  # 调整学习率
        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            best_model_weights = model.state_dict().copy()  # 深拷贝参数
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accuracies.append (val_acc)
        epochs.append(epoch)
        print(f"Epoch {epoch}:")
        print(f"  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val AUC: {val_auc:.4f}" )
    torch.save(best_model_weights, path + "train.pth")
    model.load_state_dict(best_model_weights)

    # 保存最佳模型
    data["train_losses"]=train_losses
    data["val_losses"]=val_losses
    data["val_accuracies"] = val_accuracies
    data["epochs"] = epochs
    df = pd.DataFrame.from_dict(data)
    df.to_csv(path+"log.csv", index=False)
    return model
def set_seed(seed):
    # 设置Python随机种子
    random.seed(seed)
    # 设置NumPy随机种子
    np.random.seed(seed)
    # 设置PyTorch随机种子
    torch.manual_seed(seed)
    # 如果使用CUDA（GPU），设置CUDA随机种子
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
def probe_lstm(model, tokenizer, data, input_output_ids, token, layer, probe_at, seeds,
              model_name, dataset_name, n_samples,
          data_test=None, input_output_ids_test=None, clf=None,only_train=False):
    output_file_path = f"train/{dataset_name}/"
    if not os.path.exists(output_file_path):
        os.makedirs(output_file_path)
    train_clf = True
    # Data preprocessing
    if_save = True

    sample_number = np.arange(len(data)) if n_samples == 'all' else np.arange(min(len(data), int(n_samples)))
    data_train_valid, _, input_output_ids_train_valid, _, y_train_valid, _, \
        exact_answer_train_valid, _, validity_of_exact_answer_train_valid, _, \
        questions_train_valid, _ = prepare_for_probing(
        data, input_output_ids, sample_number, [])

    if train_clf:
        output_file_path_train = output_file_path + f"train-{token}.pt"
        if os.path.isfile(output_file_path_train):
            X_train_valid = torch.load(output_file_path_train)
        else:
            X_train_valid = \
            extract_internal_reps_specific_layer_and_token(model, tokenizer, questions_train_valid,
                                                           input_output_ids_train_valid, probe_at, model_name,
                                                           layer, token, exact_answer_train_valid,
                                                           validity_of_exact_answer_train_valid
                                                           )

            torch.save(X_train_valid, output_file_path_train)


    X_test = None
    y_test = None
    if data_test is not None:
        test_data_indices = data_test.index
        # if 'exact_answer' in data:
        #     test_data_indices = test_data_indices[
        #         (data_test.iloc[test_data_indices]['valid_exact_answer'] == 1) & (
        #                     data_test.iloc[test_data_indices]['exact_answer'] != 'NO ANSWER') & (
        #                     data_test.iloc[test_data_indices]['exact_answer'].map(lambda x: type(x)) == str)]

        _, _, input_output_ids_test, _, y_test, _, \
            exact_answer_test, _, validity_of_exact_answer_test, _, \
            questions_test, _ = prepare_for_probing(
            data_test, input_output_ids_test, test_data_indices, [])
        output_file_path_test = output_file_path + f"test-{token}.pt"
        if os.path.isfile(output_file_path_test):
            X_test = torch.load(output_file_path_test)
        else:
            X_test = extract_internal_reps_specific_layer_and_token(model, tokenizer, questions_test,
                                                                input_output_ids_test, probe_at, model_name, layer,
                                                                token,
                                                                exact_answer_test, validity_of_exact_answer_test)

            torch.save(X_test, output_file_path_test)
    valid_metrics_per_seed = defaultdict(list)
    test_metrics_per_seed = defaultdict(list)

    for seed in seeds:
        print(f"##### {seed} #####")
        set_seed(seed)
        n_validation_samples = 1000 if n_samples == 'all' else min(1000, int(n_samples))
        training_data_indices, validation_data_indices = compile_probing_indices(data_train_valid, n_samples,
                                                                                 seed,
                                                                n_validation_samples=n_validation_samples)


        train_clf=False
        if train_clf:
            start_end = [0,16]
            print("atten_model")
            save_path_base = f"train/{dataset_name}/attens_models_{start_end[0]}-{start_end[1]}/"
            save_path = save_path_base+f"{seed}/"
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            # clf = init_and_train_classifier_lstm(if_save, X_train_valid, y_train_valid, save_path + f"all-")
            # clf_only_train = init_and_train_classifier_lstm(if_save, X_train_valid[training_data_indices],
            #                                                 y_train_valid[training_data_indices], save_path)


            if not only_train:
                X_test_, y_test_ = resample(X_test[test_data_indices], y_test[test_data_indices], random_state=seed)
                clf = init_and_train_classifier_lstm(if_save, X_train_valid[sample_number], y_train_valid[sample_number],save_path+f"all-",X_test_, y_test_,start_end)
                test_metrics_for_seed_all = test_classifier_lstm(clf, X_test_, y_test_,start_end=start_end)
            else:
                test_data_indices = test_data_indices[
                            (data_test.iloc[test_data_indices]['valid_exact_answer'] == 1) & (
                                        data_test.iloc[test_data_indices]['exact_answer'] != 'NO ANSWER') & (
                                        data_test.iloc[test_data_indices]['exact_answer'].map(lambda x: type(x)) == str)]

                X_test_, y_test_ = resample(X_test[test_data_indices], y_test[test_data_indices], random_state=seed)
                clf_only_train = init_and_train_classifier_lstm(if_save, X_train_valid[sample_number][training_data_indices],
                                            y_train_valid[sample_number][training_data_indices],save_path,X_test_, y_test_,start_end)
                test_metrics_for_seed_all=test_classifier_lstm(clf_only_train, X_test_, y_test_,start_end)
            test_metrics_for_seed_all["only_train"]=only_train
            for k in test_metrics_for_seed_all:
                test_metrics_per_seed[k].append(test_metrics_for_seed_all[k])
        # if data_test is not None:
        #     # pred = clf.predict(X_test)
        #     # valid_index = np.logical_xor(pred, y_test).astype(int)
        #     # data_reserve = data_test.iloc[test_data_indices][valid_index==1]
        #     # pd.DataFrame.from_dict(data_reserve).to_csv(file_path_answers+str(seed)+".csv")
        #     X_test_, y_test_ = resample(X_test, y_test, random_state=seed)
        #     # test_metrics_for_seed_all = test_classifier_lstm(clf, X_test_, y_test_)
        #     # for key in list(test_metrics_for_seed_all.keys()):  # 使用list()创建键的副本
        #     #     test_metrics_for_seed_all[key + '-all'] = test_metrics_for_seed_all.pop(key)
        #     # for k in test_metrics_for_seed_all:
        #     #     test_metrics_per_seed[k].append(test_metrics_for_seed_all[k])
        #     # test_metrics_for_seed = test_classifier_lstm(clf_only_train, X_test_, y_test_)
        #     # for k in test_metrics_for_seed:
        #     #     test_metrics_per_seed[k].append(test_metrics_for_seed[k])
        #     if not only_train:
        #         test_metrics_for_seed_all= test_classifier_lstm(clf, X_test_, y_test_)
        #         for key in list(test_metrics_for_seed_all.keys()):  # 使用list()创建键的副本
        #             test_metrics_for_seed_all[key + '-all'] = test_metrics_for_seed_all.pop(key)
        #     for k in test_metrics_for_seed_all:
        #         test_metrics_per_seed[k].append(test_metrics_for_seed_all[k])
        #     else:
        #         test_metrics_for_seed = test_classifier_lstm(clf_only_train, X_test_, y_test_)
        #         for k in test_metrics_for_seed:
        #             test_metrics_per_seed[k].append(test_metrics_for_seed[k])
        #     print("test")

    # compute mean, std per metric
    valid_metrics_aggregated = aggregate_metrics_across_seeds(valid_metrics_per_seed)
    test_metrics_aggregated = aggregate_metrics_across_seeds(test_metrics_per_seed)


    import json
    with open(f"{save_path_base}/wandb_summary.json", "w") as f:
        json.dump(test_metrics_aggregated, f, indent=4)
    print(f"Summary saved to {save_path_base}/wandb_summary.json")
    return valid_metrics_aggregated, test_metrics_aggregated, clf



def aggregate_metrics_across_seeds(metrics_per_seed):
    metrics_aggregated = {}
    for k in metrics_per_seed:
        metrics_aggregated[f"{k}"] = np.mean(metrics_per_seed[k])
        metrics_aggregated[f"{k}_std"] = np.std(metrics_per_seed[k])
    return metrics_aggregated


def init_and_train_classifier(seed, X_train, y_train):
    clf = LogisticRegression(random_state=seed).fit(X_train, y_train)
    return clf


def get_saved_clf_if_exists(args):
    if not exists("checkpoints"):
        os.makedirs("checkpoints")
    save_path = f"checkpoints/clf_{MODEL_FRIENDLY_NAMES[args.model]}_{args.dataset}_layer-{args.layer}_token-{args.token}_all_prompt.pkl"
    print("Loading classifier from ", save_path)

    if exists(save_path):
        save_clf = False
        with open(save_path, 'rb') as f:
            clf = pickle.load(f)
    else:
        save_clf = True
        clf = None
        print("Classifier not found, training new one")
    return clf, save_clf, save_path

def main():
    args = parse_args_and_init_wandb()
    model_path=LIST_OF_MODELS_PATH[args.model]
    #model, tokenizer = load_model_and_validate_gpu(model_path)
    print(model)
    data_test = None
    input_output_ids_test = None
    model_output_file = f"{args.dataset}/{MODEL_FRIENDLY_NAMES[args.model]}-answers-{args.dataset}.csv"
    data = pd.read_csv(model_output_file).reset_index()
    input_output_ids = torch.load(
        f"output/{MODEL_FRIENDLY_NAMES[args.model]}-input_output_ids-{args.dataset}.pt")
    # args.save_clf =True
    if args.test_dataset is not None:
        test_dataset = args.test_dataset
    else:
        test_dataset = args.dataset
    model_output_file_test = f"{args.dataset}/{MODEL_FRIENDLY_NAMES[args.model]}-answers-{args.dataset}_test.csv"
    load_test = False
    if os.path.isfile(model_output_file_test):
        data_test = pd.read_csv(model_output_file_test)
        input_output_ids_test = torch.load(
            f"output/{MODEL_FRIENDLY_NAMES[args.model]}-input_output_ids-{test_dataset}_test.pt")
        load_test = True

    if args.save_clf:
        clf, save_clf, save_path = get_saved_clf_if_exists(args)
    else:
        save_clf = False
        clf = None
    model = None
    tokenizer = None
    print(args.only_train)
    res = probe_lstm(model, tokenizer, data, input_output_ids, args.token,
                                                   args.layer, args.probe_at, args.seeds, args.model, args.dataset,
                                                    args.n_samples, data_test, input_output_ids_test, clf,args.only_train)



    # metrics_valid, _, clf = res
    # for m in metrics_valid:
    #     wandb.summary[m] = metrics_valid[m]
    #
    # if save_clf:
    #     with open(save_path, 'wb') as f:
    #         pickle.dump(clf, f)
    #     print("Saved classifier to ", save_path)


if __name__ == "__main__":
    main()