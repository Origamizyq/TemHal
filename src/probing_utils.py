import json
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import transformers
from baukit import TraceDict
from datasets import load_dataset
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

N_LAYERS_MISTRAL = 32
N_LAYER_LLAMA = 32

LAYERS_TO_TRACE_MISTRAL = {
    'mlp': [f"model.layers.{i}.mlp" for i in range(N_LAYERS_MISTRAL)],
    'mlp_last_layer_only': [f"model.layers.{i}.mlp.down_proj" for i in range(N_LAYERS_MISTRAL)],
    'mlp_last_layer_only_input': [f"model.layers.{i}.mlp.down_proj" for i in range(N_LAYERS_MISTRAL)],
    'attention_heads': [f"model.layers.{i}.self_attn.o_proj" for i in range(N_LAYERS_MISTRAL)],
    'attention_output': [f"model.layers.{i}.self_attn.o_proj" for i in range(N_LAYERS_MISTRAL)],
}

LAYERS_TO_TRACE_LLAMA = {
    'mlp': [f"model.layers.{i}.mlp" for i in range(N_LAYER_LLAMA)],
    'mlp_last_layer_only': [f"model.layers.{i}.mlp.down_proj" for i in range(N_LAYER_LLAMA)],
    'mlp_last_layer_only_input': [f"model.layers.{i}.mlp.down_proj" for i in range(N_LAYER_LLAMA)],
    'attention_heads': [f"model.layers.{i}.self_attn.o_proj" for i in range(N_LAYER_LLAMA)],
    'attention_output': [f"model.layers.{i}.self_attn.o_proj" for i in range(N_LAYER_LLAMA)],
}

LAYERS_TO_TRACE = {
    'mistralai/Mistral-7B-Instruct-v0.2': LAYERS_TO_TRACE_MISTRAL,
    'mistralai/Mistral-7B-v0.3': LAYERS_TO_TRACE_MISTRAL,
    'meta-llama/Meta-Llama-3-8B-Instruct': LAYERS_TO_TRACE_LLAMA,
    'meta-llama/Meta-Llama-3-8B': LAYERS_TO_TRACE_LLAMA,
}

N_LAYERS = {
    'mistralai/Mistral-7B-Instruct-v0.2': N_LAYERS_MISTRAL,
    'mistralai/Mistral-7B-v0.3': N_LAYERS_MISTRAL,
    'meta-llama/Meta-Llama-3-8B-Instruct': N_LAYER_LLAMA,
    'meta-llama/Meta-Llama-3-8B': N_LAYER_LLAMA,
}

HIDDEN_SIZE = {
    'tiiuae/falcon-40b-instruct': 8192,
    'mistralai/Mistral-7B-Instruct-v0.2': 4096,
    'mistralai/Mistral-7B-v0.3': 4096,
    'meta-llama/Meta-Llama-3-8B-Instruct': 8192,
    'meta-llama/Meta-Llama-3-8B': 8192,
    'google/gemma-7b': 3072,
    'google/gemma-7b-it': 3072,
}

LIST_OF_DATASETS = ['triviaqa',
                    'triviaqa_test',
                    'imdb',
                    'winobias',
                    'hotpotqa',
                    'hotpotqa_test',
                    'hotpotqa_with_context',
                    'math',
                    'movies',
                    'movies_test',
                    'mnli',
                    'mnli_test',
                    'natural_questions_with_context',
                    'winogrande',
                    'winogrande_test',
                    'math_test'
                    ]

LIST_OF_TEST_DATASETS = [f"{x}_test" for x in LIST_OF_DATASETS]

LIST_OF_MODELS = ['mistralai/Mistral-7B-Instruct-v0.2',
                                            'mistralai/Mistral-7B-v0.3',
                                            'meta-llama/Meta-Llama-3-8B',
                                            'meta-llama/Meta-Llama-3-8B-Instruct',
                                            ]
LIST_OF_MODELS_PATH={
    'meta-llama/Meta-Llama-3-8B-Instruct':'/GLOBALFS/nudt_dwfeng_1/zyq/data/models/LLM-Research/Meta-Llama-3-8B-Instruct',
    'mistralai/Mistral-7B-Instruct-v0.2':'/GLOBALFS/nudt_dwfeng_1/zyq/data/models/AI-ModelScope/Mistral-7B-Instruct-v0.2'
}
MODEL_FRIENDLY_NAMES = {
    'mistralai/Mistral-7B-Instruct-v0.2': 'mistral-7b-instruct',
    'mistralai/Mistral-7B-v0.3': 'mistral-7b',
    'meta-llama/Meta-Llama-3-8B': 'llama-3-8b',
    'meta-llama/Meta-Llama-3-8B-Instruct': 'llama-3-8b-instruct',
}

LIST_OF_PROBING_LOCATIONS = ['mlp', 'mlp_last_layer_only', 'mlp_last_layer_only_input', 'attention_output']


def encode(prompt, tokenizer, model_name):
    messages = [
        {"role": "user", "content": prompt}
    ]
    model_input = tokenizer.apply_chat_template(messages, return_tensors="pt")[0]
    return model_input


def tokenize(prompt, tokenizer, model_name, tokenizer_args=None):
    if 'instruct' in model_name.lower():
        messages = [
            {"role": "user", "content": prompt}
        ]
        model_input = tokenizer.apply_chat_template(messages, return_tensors="pt", **(tokenizer_args or {})).to('cuda')
    else: # non instruct model
        model_input = tokenizer(prompt, return_tensors='pt', **(tokenizer_args or {}))
        if "input_ids" in model_input:
            model_input = model_input["input_ids"].to('cuda')
    return model_input


def generate(model_input, model, model_name, do_sample=False, output_scores=False, temperature=1.0, top_k=50, top_p=1.0,
             max_new_tokens=100, stop_token_id=None, tokenizer=None, output_hidden_states=False, additional_kwargs=None):

    if stop_token_id is not None:
        eos_token_id = stop_token_id
    else:
        eos_token_id = tokenizer.eos_token_id

    model_output = model.generate(model_input,
                                  max_new_tokens=max_new_tokens, output_hidden_states=output_hidden_states,
                                  output_scores=output_scores,
                                  return_dict_in_generate=True, do_sample=do_sample,
                                  temperature=temperature, top_k=top_k, top_p=top_p, eos_token_id=eos_token_id,
                                  **(additional_kwargs or {}))

    return model_output

def get_indices_of_exact_answer(tokenizer, input_output_ids, exact_answer, model_name, prompt=None, output_ids=None):

    if output_ids is not None:
        lower = input_output_ids.shape[0] - output_ids.shape[0]
    elif prompt is not None:
        prompt_len = tokenize(prompt, tokenizer, model_name).shape[1]
        lower = prompt_len
    else:
        lower = 1

    full_question_answer = tokenizer.decode(input_output_ids[lower:])
    exact_answer_index = full_question_answer.lower().find(exact_answer.lower().strip())

    if exact_answer_index == -1:
        print("############ ERROR")
        print(exact_answer, "#", full_question_answer)
    assert(exact_answer_index != -1)
    true_exact_answer = full_question_answer[exact_answer_index:exact_answer_index + len(exact_answer)]
    assert true_exact_answer in full_question_answer

    higher = len(input_output_ids) - 1

    while true_exact_answer in tokenizer.decode(input_output_ids[lower:higher + 1]):
        higher -= 1
    higher += 1
    while true_exact_answer in tokenizer.decode(input_output_ids[lower:higher + 1]):
        lower += 1
    lower -= 1

    return list(range(lower, higher + 1))

def exact_answer_is_valid(exact_answer_valid, exact_answer):
    return (exact_answer_valid == 1) and (exact_answer != 'NO ANSWER') and (type(exact_answer) == str) and (
                len(exact_answer) > 0)

# A reusable dictionary in case we want to extract the exact answer from the same answer several times during a run
# for efficiency
exact_tokens_dict = {}
def get_token_index(token, tokenizer, question, model_name, full_answer_tokenized=None, exact_answer=None,
                    exact_answer_valid=None, use_dict=True):

    if (type(token) == str) and ('exact' in token):
        if exact_answer_is_valid(exact_answer_valid, exact_answer):
            if (not use_dict) or (question not in exact_tokens_dict):
                t = get_indices_of_exact_answer(tokenizer, full_answer_tokenized, exact_answer, model_name, prompt=question)
                exact_tokens_dict[question] = t
            else:
                t = exact_tokens_dict[question]

            if token == 'exact_answer_last_token':
                t = min(len(full_answer_tokenized) - 1, t[-1])
            elif token == 'exact_answer_first_token':
                t = t[0]
            elif token == 'exact_answer_before_first_token':
                t = t[0] - 1
            elif token == 'exact_answer_after_last_token':
                t = min(len(full_answer_tokenized) - 1, t[-1] + 1)
        else:
            t = get_token_index('last_q_token', tokenizer, question, model_name, exact_answer, exact_answer_valid) # default case. In the paper we're not supposed to get here.
    else:
        q_length = len(tokenize(question, tokenizer, model_name)[0])
        if token == 'last_q_token':
            t = q_length - 1
        elif token == 'first_answer_token':
            t = q_length
        elif token == 'second_answer_token':
            t = q_length + 1
        else:
            try:
                token = int(token)
            except ValueError:
                pass
            t = token
    return t


def get_embeddings_in_token(token, layer, extracted_embeddings, tokenizer, prompts, model_name,
                            full_answers_tokenized=None, exact_answers=None, valid_exact_answers=None,
                            use_dict=True):
    X = []
    for idx in range(len(prompts)):

        if (full_answers_tokenized is not None) and (exact_answers is not None) and (valid_exact_answers is not None):
            t = get_token_index(token, tokenizer, prompts[idx], model_name, full_answers_tokenized[idx],
                                exact_answers[idx], valid_exact_answers[idx], use_dict=use_dict)
        else:
            t = get_token_index(token, tokenizer, prompts[idx], model_name, use_dict=use_dict)

        if layer == 'all':
            X.append(extracted_embeddings[idx][:, t].float().numpy())
        else:
            X.append(extracted_embeddings[idx][layer][t].float().numpy())
    return X


def extract_internal_reps_single_sample(model, model_input, probe_at, model_name):

    model_input = model_input.to(model.device)
    layers_to_trace = get_probing_layer_names(probe_at, model_name)

    with torch.no_grad():
        with TraceDict(model, layers_to_trace, retain_input=True, clone=True) as ret:
            output = model(model_input.unsqueeze(dim=0), output_hidden_states=True)

    if 'attention' in probe_at:
        output_per_layer = get_attention_output(model, ret, layers_to_trace, probe_at)
    elif 'mlp' in probe_at:
        output_per_layer = get_mlp_output(ret, layers_to_trace, probe_at)
    else:
        raise TypeError("Probe type not supported")

    return output_per_layer


def get_mlp_output(ret, layers_to_trace, probe_at):
    mlp_output_per_layer = []
    mlp_input_per_layer = []
    for k in layers_to_trace:
        mlp_output_per_token = ret[k].output.squeeze().cpu()
        mlp_output_per_layer.append(mlp_output_per_token)
        mlp_input_per_token = ret[k].input.squeeze().cpu()
        mlp_input_per_layer.append(mlp_input_per_token)

    if 'input' in probe_at:
        return mlp_input_per_layer
    else:
        return mlp_output_per_layer


def get_attention_output(model, ret, layers_to_trace, probe_at):
    attention_output_per_layer = []
    for k in layers_to_trace:
        heads_per_token = ret[k].output.reshape(ret[k].input.shape[0],
                                                ret[k].input.shape[1],
                                                model.model.layers[0].self_attn.num_heads,
                                                model.model.layers[0].self_attn.head_dim).transpose(1, 2)
        attention_output = ret[k].output.squeeze().cpu()
        attention_output_per_layer.append(attention_output)


    return attention_output_per_layer
# def get_token_index_of_sentence_by_js(model,output,token, tokenizer, question, model_name, full_answer_tokenized=None, exact_answer=None,
#                     exact_answer_valid=None, use_dict=True):
#     if (type(token) == str) and ('exact' in token):
#         if exact_answer_is_valid(exact_answer_valid, exact_answer):
#             if (not use_dict) or (question not in exact_tokens_dict):
#                 t = get_indices_of_exact_answer(tokenizer, full_answer_tokenized, exact_answer, model_name, prompt=question)
#                 exact_tokens_dict[question] = t
#             else:
#                 t = exact_tokens_dict[question]
#     last_index = None
#     value = 128007
#     indices = torch.where(full_answer_tokenized==value)[0]
#     if len(indices) > 0:
#         last_index = indices.max().item()
#     else:
#         last_index = -1
#     # messages = [17666 277
#     #     {"role": "user", "content": question}
#     # ]
#     # model_input = tokenizer.apply_chat_template(messages, return_tensors="pt").to('cuda')
#     #
#     from transformers.generation.utils import LogitsProcessorList
#     logits_processor = LogitsProcessorList()
#     output_hidden = model(full_answer_tokenized.unsqueeze(0).cuda(0),return_dict=True,output_hidden_states=True).hidden_states
#     lm_head = model.get_output_embeddings()
#     norm = model.model.norm
#
#
#     tokens = tokenizer.convert_ids_to_tokens(full_answer_tokenized)
#     key_words_norm_2 = []
#     other_norm_2 = []
#     mature_logits =lm_head(norm(output_hidden[-1].cuda(0)))
#     mature_logits =logits_processor(full_answer_tokenized, mature_logits)
#     output_csv = {}
#     for k in range(last_index,output[0].shape[0]):
#         temp = tokens[k]
#         output_csv[temp] = []
#     js_divs_list_layer=[]
#     js_divs_list_layer.append(tokens[last_index:output[0].shape[0]])
#     for i in range(1,32):
#         temp_logits = lm_head(norm(output_hidden[i].cuda(0)))
#         temp_logits=logits_processor(full_answer_tokenized, temp_logits)
#         js_divs_list = []
#         for j in range(last_index,output[0].shape[0]):
#             token_temp_logit = temp_logits[0][j]
#             token_mature_logit = mature_logits[0][j]
#             top_kmature_scores, top_k_indices = torch.topk(token_mature_logit, 100)
#             top_kmature_scores = torch.tensor([1.0,1.0,5.0,6.0,7.0,8.0])
#             top_ktemp_scores = torch.tensor([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
#             # top_ktemp_scores=token_temp_logit[top_k_indices]
#
#             softmax_mature_layer = F.softmax(top_kmature_scores,
#                                              dim=-1)  # shape: (batch_size, num_features)
#             softmax_premature_layers = F.softmax(top_ktemp_scores,
#                                                  dim=-1)  # shape: (num_premature_layers, batch_size, num_features)
#             #  softmax_mature_layers = softmax_mature_layer.unsqueeze(0)
#             #  softmax_get_layers = torch.cat((softmax_premature_layers,softmax_mature_layers),dim=0)
#             # 3. Calculate M, the average distribution
#             M = 0.5 * (softmax_mature_layer+ softmax_premature_layers)  # shape: (num_premature_layers, batch_size, num_features)
#
#             # 4. Calculate log-softmax for the KL divergence
#             log_softmax_mature_layer = F.log_softmax(top_kmature_scores,
#                                                      dim=-1)  # shape: (batch_size, num_features)
#             log_softmax_premature_layers = F.log_softmax(top_ktemp_scores,
#                                                          dim=-1)  # shape: (num_premature_layers, batch_size, num_features)
#
#             # 5. Calculate the KL divergences and then the JS divergences
#             kl1 = F.kl_div(log_softmax_mature_layer, M, reduction='none').mean(
#                 -1)  # shape: (num_premature_layers, batch_size)
#             kl2 = F.kl_div(log_softmax_premature_layers, M, reduction='none').mean(
#                 -1)  # shape: (num_premature_layers, batch_size)
#             js_divs = 0.5 * (kl1 + kl2)  # shape: (num_premature_layers, batch_size)
#
#             # 6. Reduce the batchmean
#             js_divs = js_divs.mean(-1)
#             js_divs_list.append(js_divs.cpu().item())
#         js_divs_list_layer.append(js_divs_list)
#     import csv
#     with open("1.csv","w",newline="") as f:
#         writer = csv.writer(f)
#         writer.writerows(js_divs_list_layer)
#     print("test")
#     print(f"lsyer:{i}key js: {np.mean(key_words_norm_2)},other_js: {np.mean(other_norm_2)}")
#     print("its all done")


def extract_internal_reps_specific_layer_and_token(model, tokenizer, prompts, input_output_ids_lst,
                                                   probe_at, model_name, layer, token, exact_answers,
                                                   exact_answers_valid, use_dict_for_tokens=False):
    all_reps = []
    length = len(input_output_ids_lst)
    print(
        f"Extracting internal reps from layer {layer} and token {token} from {length} textual inputs...")
    rep_tensor = None
    for idx, (input_output_ids, prompt, exact_answer, exact_answer_valid) in tqdm(enumerate(zip(input_output_ids_lst, prompts, exact_answers, exact_answers_valid))):

        output = extract_internal_reps_single_sample(model, input_output_ids, probe_at, model_name)
        # t=get_token_index_of_sentence_by_js(model,output,token, tokenizer, prompt, model_name, input_output_ids,
        #                      exact_answer, exact_answer_valid, use_dict=use_dict_for_tokens)
        t = get_token_index(token, tokenizer, prompt, model_name, input_output_ids,
                            exact_answer, exact_answer_valid, use_dict=use_dict_for_tokens)

        # rep = output[layer][t].float().numpy()

        rep_list=[]
        for i in output:
            rep_list.append(i[t])
        temp = torch.stack(rep_list).unsqueeze(0)
        if rep_tensor is None:
            rep_tensor = temp
        else:
            rep_tensor = torch.cat([rep_tensor, temp], dim=0)

    return rep_tensor


def extract_internal_reps_all_layers_and_tokens(model, input_output_ids_lst, probe_at, model_name):
    all_outputs_per_layer = []

    length = len(input_output_ids_lst)
    print(f"Extracting internal reps from {length} textual inputs...")

    for input_output_ids in tqdm(input_output_ids_lst):
        output = extract_internal_reps_single_sample(model, input_output_ids, probe_at, model_name)

        all_outputs_per_layer.append(output)

    return all_outputs_per_layer


def load_model_and_validate_gpu(model_path, tokenizer_path=None):
    if tokenizer_path is None:
        tokenizer_path = model_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    print("Started loading model")
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map='auto',
                                                 torch_dtype=torch.bfloat16)
    assert ('cpu' not in model.hf_device_map.values())
    return model, tokenizer
def test_classifier_lstm(model, X_test, y_test, pos_label=0, predicted_probas=None,start_end=[]):
    y_valid=torch.FloatTensor(y_test.reshape(-1, 1))
    X_valid=X_test[:,start_end[0]:start_end[1]].float().to("cuda")
    model.eval()
    if predicted_probas is None:
        baseline_acc = max(y_valid.mean(), (1-y_valid).mean())
        pred = model.predict(X_valid).detach().cpu()
        acc = (pred == y_valid).float().mean()
        acc_diff_from_baseline = acc - baseline_acc
        precision = precision_score(y_valid, pred)
        recall = recall_score(y_valid, pred)
        f1 = f1_score(y_valid, pred)
        predicted_probas = model(X_valid).detach().cpu()
    else:
        baseline_acc = None
        acc = None
        acc_diff_from_baseline = None
        precision = None
        recall = None
        f1 = None

    fpr_for_auc, tpr_for_auc, thresholds = metrics.roc_curve(y_valid.detach(), torch.sigmoid(predicted_probas))
    auc = metrics.auc(fpr_for_auc, tpr_for_auc)
    print("acc_diff_from_baseline: {},f1: {}, precision: {}, recall: {}, auc: {}, baseline_acc: {}, acc: {}".format(
        acc_diff_from_baseline, f1, precision, recall, auc, baseline_acc,
        acc))
    return {"acc_diff_from_baseline": acc_diff_from_baseline.tolist(), "f1": f1.tolist(), "precision": precision.tolist(), "recall": recall.tolist(),
            "auc": auc.tolist(), "baseline_acc": baseline_acc.tolist(), "acc": acc.tolist()}

def compute_metrics_probing(clf, X_valid, y_valid, pos_label=0, predicted_probas=None):
    pos_label=1
    if predicted_probas is None:
        baseline_acc = max(y_valid.mean(), (1-y_valid).mean())
        pred = clf.predict(X_valid)
        acc = (pred == y_valid).mean()
        acc_diff_from_baseline = acc - baseline_acc
        precision = precision_score(y_valid, pred, pos_label=pos_label)
        recall = recall_score(y_valid, pred, pos_label=pos_label)
        f1 = f1_score(y_valid, pred, pos_label=pos_label)
        predicted_probas = clf.predict_proba(X_valid)
        predicted_probas = predicted_probas[:, pos_label]
    else:
        baseline_acc = None
        acc = None
        acc_diff_from_baseline = None
        precision = None
        recall = None
        f1 = None

    fpr_for_auc, tpr_for_auc, thresholds = metrics.roc_curve(y_valid, predicted_probas, pos_label=pos_label)
    auc = metrics.auc(fpr_for_auc, tpr_for_auc)
    print("acc_diff_from_baseline: {},f1: {}, precision: {}, recall: {}, auc: {}, baseline_acc: {}, acc: {}".format(
        acc_diff_from_baseline, f1, precision, recall, auc, baseline_acc,
        acc))
    return {"acc_diff_from_baseline": acc_diff_from_baseline.tolist(), "f1": f1.tolist(), "precision": precision.tolist(), "recall": recall.tolist(),
            "auc": auc.tolist(), "baseline_acc": baseline_acc.tolist(), "acc": acc.tolist()}


def probe_specific_layer_token(extracted_embeddings_train, extracted_embeddings_valid, layer, token, questions_train,
                               questions_valid, full_answer_tokenized_train, full_answer_tokenized_valid,
                               exact_answer_train, exact_answer_valid, validity_exact_answer_train,
                               validity_exact_answer_valid,
                               tokenizer, y_train, y_valid, seed, model_name,
                               use_dict_for_tokens=True):

    X_train = get_embeddings_in_token(token, layer, extracted_embeddings_train, tokenizer,
                                      questions_train, model_name, full_answer_tokenized_train, exact_answer_train,
                                      validity_exact_answer_train, use_dict=use_dict_for_tokens)
    X_valid = get_embeddings_in_token(token, layer, extracted_embeddings_valid, tokenizer,
                                      questions_valid, model_name, full_answer_tokenized_valid, exact_answer_valid,
                                      validity_exact_answer_valid,
                                      use_dict=use_dict_for_tokens)

    clf = LogisticRegression(random_state=seed).fit(X_train, y_train)

    return compute_metrics_probing(clf, X_valid, y_valid, pos_label=0)


def compile_probing_indices(data, n_samples, seed, n_validation_samples=0):

    n_samples = eval(n_samples)
    indices = np.arange(len(data))

    if n_validation_samples > 0:
        n_validation_samples = min(n_validation_samples, round(0.2 * (len(indices))))
        indices, validation_data_indices = train_test_split(indices, test_size=n_validation_samples, random_state=seed)

    if n_samples != 'all' and type(n_samples) == int:
        np.random.shuffle(indices)
        indices = indices[:n_samples]  # should be consistent across runs same seed

    if n_validation_samples > 0:
        training_data_indices = indices
    else:
        training_data_indices, validation_data_indices = train_test_split(indices, test_size=0.2, random_state=seed)

    if 'exact_answer' in data:
        training_data_indices = training_data_indices[(data.iloc[training_data_indices]['valid_exact_answer'] == 1) & (data.iloc[training_data_indices]['exact_answer'] != 'NO ANSWER') & (data.iloc[training_data_indices]['exact_answer'].map(lambda x : type(x)) == str)]
        validation_data_indices = validation_data_indices[(data.iloc[validation_data_indices]['valid_exact_answer'] == 1) & (data.iloc[validation_data_indices]['exact_answer'] != 'NO ANSWER') & (data.iloc[validation_data_indices]['exact_answer'].map(lambda x : type(x)) == str)]

    return training_data_indices, validation_data_indices


def get_probing_layer_names(probe_at, model_name):
    if probe_at in ['mlp_last_layer_only', 'mlp_last_layer_only_input']:
        probe_at = 'mlp'
    layers_to_trace = LAYERS_TO_TRACE[model_name][probe_at]
    return layers_to_trace


def prepare_for_probing(data, input_output_ids, training_data_indices, validation_data_indices):

    # small fixture to verify input is not too large which may cause memory overload
    training_data_indices = [i for i in training_data_indices if len(input_output_ids[i]) <= 10000]
    validation_data_indices = [i for i in validation_data_indices if len(input_output_ids[i]) <= 10000]

    data_train = data.iloc[training_data_indices].reset_index()
    data_valid = data.iloc[validation_data_indices].reset_index()


    y_train = data_train['automatic_correctness'].to_numpy()
    y_valid = data_valid['automatic_correctness'].to_numpy()

    input_output_ids_train = [input_output_ids[i] for i in training_data_indices]
    input_output_ids_valid = [input_output_ids[i] for i in validation_data_indices]

    if 'exact_answer' in data:
        exact_answer_train = data_train['exact_answer']
        exact_answer_valid = data_valid['exact_answer']
        validity_of_exact_answer_train = data_train['valid_exact_answer'].astype(int)
        validity_of_exact_answer_valid = data_valid['valid_exact_answer'].astype(int)
    else:
        exact_answer_train = None
        exact_answer_valid = None
        validity_of_exact_answer_train = None
        validity_of_exact_answer_valid = None

    questions_train = data.iloc[training_data_indices].reset_index()['question']
    questions_valid = data.iloc[validation_data_indices].reset_index()['question']

    return data_train, data_valid, input_output_ids_train, input_output_ids_valid, y_train, y_valid,\
            exact_answer_train, exact_answer_valid, validity_of_exact_answer_train, validity_of_exact_answer_valid, \
            questions_train, questions_valid
