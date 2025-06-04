def get_token_index_of_sentence_by_js(model,output,token, tokenizer, question, model_name, full_answer_tokenized=None, exact_answer=None,
                    exact_answer_valid=None, use_dict=True):
    if (type(token) == str) and ('exact' in token):
        if exact_answer_is_valid(exact_answer_valid, exact_answer):
            if (not use_dict) or (question not in exact_tokens_dict):
                t = get_indices_of_exact_answer(tokenizer, full_answer_tokenized, exact_answer, model_name, prompt=question)
                exact_tokens_dict[question] = t
            else:
                t = exact_tokens_dict[question]
    last_index = None
    value = 128007
    indices = torch.where(full_answer_tokenized==value)[0]
    if len(indices) > 0:
        last_index = indices.max().item()
    else:
        last_index = -1
    # messages = [17666 277
    #     {"role": "user", "content": question}
    # ]
    # model_input = tokenizer.apply_chat_template(messages, return_tensors="pt").to('cuda')
    #
    output_hidden = model(full_answer_tokenized.unsqueeze(0).cuda(0),return_dict=True,output_hidden_states=True).hidden_states
    lm_head = model.get_output_embeddings()
    norm = model.model.norm
    last_mature = lm_head(output_hidden[-1].cuda(0)[0])
    softmax_mature_layer = torch.softmax(last_mature, dim=-1)
    log_softmax_mature_layer=F.log_softmax(last_mature, dim=-1)

    token_layer_dict = {}
    tokens = tokenizer.convert_ids_to_tokens(full_answer_tokenized)
    select_token_norm_dict = {}
    for i in range(31,len(output_hidden)-1):
        temp_logits = lm_head(output_hidden[i].cuda(0))[0]
        temp_logits_norm = lm_head(norm(output_hidden[i].cuda(0)))[0]
        temp_fix_logits = lm_head(output_hidden[31].cuda(0))[0]
        temp_fix_logits_norm = lm_head(norm(output_hidden[31].cuda(0)))[0]
        other_js = []
        key_words_js = []
        all_js =[]
        select_token_norm = []
        select_token_norm_js = []
        for j in range(last_index+1,output[0].shape[0]-1):
            softmax_premature_layer = torch.softmax(temp_fix_logits_norm[j], dim=-1)

            log_softmax_premature_layer = F.log_softmax(temp_fix_logits_norm[j], dim=-1)
            # softmax_premature_layers = torch.softmax(temp_logits[j], dim=-1)
            avg_dist = 0.5 * (softmax_mature_layer[j] + softmax_premature_layer)
            # shape: (num_premature_layers, batch_size, vocab_size)
            # log_softmax_premature_layers = F.log_softmax(temp_logits[j], dim=-1)

            # 5. Calculate the KL divergences and then the JS divergences
            # shape: (num_premature_layers, batch_size)
            kl1 = F.kl_div(log_softmax_mature_layer[j], avg_dist, reduction="none").mean(-1)
            # shape: (num_premature_layers, batch_size)
            kl2 = F.kl_div(log_softmax_premature_layer, avg_dist, reduction="none").mean(-1)
            js_divs = 0.5 * (kl1 + kl2)
            # 6. Reduce the batchmean
            js_divs = js_divs.mean(-1)*10000
            js_divs = float(js_divs.cpu().detach())
            all_js.append(js_divs)
            if j in t:
                key_words_js.append(js_divs)
            else:
                other_js.append(js_divs)
            if torch.argmax(torch.softmax(temp_fix_logits_norm[j], dim=-1))!=full_answer_tokenized[j+1] and js_divs>2:
                select_token_norm.append(tokenizer.convert_ids_to_tokens([full_answer_tokenized[j+1]]))
                select_token_norm_js.append(js_divs)
        token_layer_dict[i] = all_js
        print("test")
        print(f"lsyer:{i}key js: {np.mean(key_words_js)},other_js: {np.mean(other_js)}")
    print("its all done")