from tqdm import tqdm
from data_preprocess import preprocess_data, get_webpage
import time
import selenium
from TIE_german_evaluation.src.evaluate_german import to_list
import streamlit as st
from transformers import PreTrainedModel
import TIE_german_evaluation as tie
from TIE_german_evaluation.src.model import TIEConfig, TIE
import torch
from TIE_german_evaluation.markuplmft.models.markuplm import MarkupLMConfig, MarkupLMTokenizer, MarkupLMForQuestionAnswering
from TIE_german_evaluation.src.utils import TIEExample, convert_examples_to_features, RawResult, RawTagResult, _get_best_indexes, _compute_softmax, get_final_text, write_tag_predictions
import bs4
from TIE_german_evaluation.markuplmft.data.tag_utils import tags_dict
from TIE_german_evaluation.src.dataset import StrucDataset
import os
from torch.utils.data import DataLoader, SequentialSampler, DistributedSampler
import timeit
import collections
import json





# helper methods

def html_escape(html):
    r"""
    replace the special expressions in the html file for specific punctuation.
    """
    html = html.replace('&quot;', '"')
    html = html.replace('&amp;', '&')
    html = html.replace('&lt;', '<')
    html = html.replace('&gt;', '>')
    html = html.replace('&nbsp;', ' ')
    return html  

def fetch_html(url):
    return get_webpage(url)  

def write_html(driver):  # string
    html = preprocess_data(driver)
    
    with open ('page.html', 'w') as f:
        f.write(html)
    return html

def read_html():
    with open('page.html','r') as f:
        return f.read()

def is_whitespace(c):
            if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
                return True
            return False

def html_to_text_list(h):
    tag_num, text_list = 0, []
    for element in h.descendants:
        if (type(element) == bs4.element.NavigableString) and (element.strip()):
            text_list.append(element.strip())
        if type(element) == bs4.element.Tag:
            tag_num += 1
    return text_list, tag_num + 2  # + 2 because we treat the additional 'yes' and 'no' as two special tags.

def html_to_text(h):
    tag_list = set()
    for element in h.descendants:
        if type(element) == bs4.element.Tag:
            element.attrs = {}
            temp = str(element).split()
            tag_list.add(temp[0])
            tag_list.add(temp[-1])
    return html_escape(str(h)), tag_list

def adjust_offset(offset, text):
    text_list = text.split()
    cnt, adjustment = 0, []
    for t in text_list:
        if not t:
            continue
        if t[0] == '<' and t[-1] == '>':
            adjustment.append(offset.index(cnt))
        else:
            cnt += 1
    add = 0
    adjustment.append(len(offset))
    for i in range(len(offset)):
        while i >= adjustment[add]:
            add += 1
        offset[i] += add
    return offset

def e_id_to_t_id(e_id, html):
    t_id, real_eid = 0, 0
    for element in html.descendants:
        if type(element) == bs4.element.NavigableString and element.strip():
            t_id += 1
        if type(element) == bs4.element.Tag:
            if int(element.attrs['tid']) >= e_id:
                break
            else:
                real_eid += 1
    return t_id, real_eid

def calc_num_from_raw_text_list(t_id, l):
    n_char = 0
    for i in range(t_id):
        n_char += len(l[i]) + 1
    return n_char

def word_to_tag_from_text(tokens, h):
    cnt, path = -1, []
    w2t, t2w, tags = [], [], []
    for ind in range(len(tokens) - 2):
        t = tokens[ind]
        if len(t) < 2:
            w2t.append(path[-1])
            continue
        if t[0] == '<' and t[-2] == '/':
            cnt += 1
            w2t.append(cnt)
            tags.append(t)
            t2w.append({'start': ind, 'end': ind})
            continue
        if t[0] == '<' and t[1] != '/':
            cnt += 1
            path.append(cnt)
            tags.append(t)
            t2w.append({'start': ind})
        w2t.append(path[-1])
        if t[0] == '<' and t[1] == '/':
            num = path.pop()
            t2w[num]['end'] = ind
    w2t.append(cnt + 1)
    w2t.append(cnt + 2)
    tags.append('<no>')
    tags.append('<yes>')
    t2w.append({'start': len(tokens) - 2, 'end': len(tokens) - 2})
    t2w.append({'start': len(tokens) - 1, 'end': len(tokens) - 1})
    assert len(w2t) == len(tokens)
    assert len(tags) == len(t2w), (len(tags), len(t2w))
    assert len(path) == 0, h
    return w2t, t2w, tags

def word_tag_offset(html):
    w_t, t_w, tags, tags_tids = [], [], [], []
    for element in html.descendants:
        if type(element) == bs4.element.Tag:
            content = ' '.join(list(element.strings)).split()
            t_w.append({'start': len(w_t), 'len': len(content)})
            tags.append('<' + element.name + '>')
            tags_tids.append(element['tid'])
        elif type(element) == bs4.element.NavigableString and element.strip():
            text = element.split()
            tid = element.parent['tid']
            ind = tags_tids.index(tid)
            for _ in text:
                w_t.append(ind)
    t_w.append({'start': len(w_t), 'len': 1})
    t_w.append({'start': len(w_t) + 1, 'len': 1})
    w_t.append(len(tags))
    w_t.append(len(tags) + 1)
    tags.append('<no>')
    tags.append('<yes>')
    return w_t, t_w, tags

def subtoken_tag_offset(html, s_tok, tok_s):
    w_t, t_w, tags = word_tag_offset(html)
    s_t, t_s = [], []
    unique_tids = set(range(len(tags)))
    for i in range(len(s_tok)):
        s_t.append(w_t[s_tok[i]])
    for i in t_w:
        try:
            t_s.append({'start': tok_s[i['start']], 'end': tok_s[i['start'] + i['len']] - 1})
        except IndexError:
            assert i == t_w[-1]
            t_s.append({'start': tok_s[i['start']], 'end': len(s_tok) - 1})
    return s_t, t_s, tags, unique_tids

def calculate_depth(html_code):
    def _calc_depth(tag, depth):
        for t in tag.contents:
            if type(t) != bs4.element.Tag:
                continue
            tag_depth.append(depth)
            _calc_depth(t, depth + 1)

    tag_depth = []
    _calc_depth(html_code, 1)
    tag_depth += [2, 2]
    return tag_depth

    
    
def get_xpath_and_treeid4tokens(html_code, unique_tids, max_depth):
    unknown_tag_id = len(tags_dict)
    pad_tag_id = unknown_tag_id + 1
    max_width = 1000
    width_pad_id = 1001

    pad_x_tag_seq = [pad_tag_id] * max_depth
    pad_x_subs_seq = [width_pad_id] * max_depth

    def xpath_soup(element):

        xpath_tags = []
        xpath_subscripts = []
        tree_index = []
        child = element if element.name else element.parent
        for parent in child.parents:  # type: bs4.element.Tag
            siblings = parent.find_all(child.name, recursive=False)
            para_siblings = parent.find_all(True, recursive=False)
            xpath_tags.append(child.name)
            xpath_subscripts.append(
                0 if 1 == len(siblings) else next(i for i, s in enumerate(siblings, 1) if s is child))

            tree_index.append(next(i for i, s in enumerate(para_siblings, 0) if s is child))
            child = parent
        xpath_tags.reverse()
        xpath_subscripts.reverse()
        tree_index.reverse()
        
        return xpath_tags, xpath_subscripts, tree_index
    xpath_tag_map = {}
    xpath_subs_map = {}

    for tid in unique_tids:
        element = html_code.find(attrs={'tid': tid})
        if element is None:
            xpath_tags = pad_x_tag_seq
            xpath_subscripts = pad_x_subs_seq

            xpath_tag_map[tid] = xpath_tags
            xpath_subs_map[tid] = xpath_subscripts
            continue

        xpath_tags, xpath_subscripts, tree_index = xpath_soup(element)

        assert len(xpath_tags) == len(xpath_subscripts)
        assert len(xpath_tags) == len(tree_index)

        if len(xpath_tags) > max_depth:
            xpath_tags = xpath_tags[-max_depth:]
            xpath_subscripts = xpath_subscripts[-max_depth:]

        xpath_tags = [tags_dict.get(name, unknown_tag_id) for name in xpath_tags]
        xpath_subscripts = [min(i, max_width) for i in xpath_subscripts]

        # we do not append them to max depth here

        xpath_tags += [pad_tag_id] * (max_depth - len(xpath_tags))
        xpath_subscripts += [width_pad_id] * (max_depth - len(xpath_subscripts))

        xpath_tag_map[tid] = xpath_tags
        xpath_subs_map[tid] = xpath_subscripts

    return xpath_tag_map, xpath_subs_map

class JobMonitor():
    

    def __init__(self):
        self.stage = 1
        self.model, self.tokenizer = self.build_model_and_tokenizer()
        

    def build_model_and_tokenizer(self):
        
        if self.stage == 1:
            tie_config = TIEConfig.from_pretrained("TIE_german_evaluation/result/TIE_MarkupLM/config.json")
            model = TIE.from_pretrained("TIE_german_evaluation/result/TIE_MarkupLM/pytorch_model.bin", config=tie_config)
            model.to(torch.device("cpu"))
            tokenizer = MarkupLMTokenizer.from_pretrained("TIE_german_evaluation/result/TIE_MarkupLM/", do_lower_case=False)
        
        else:
            config = MarkupLMConfig.from_pretrained("/Users/jostgotte/Documents/Uni/WS2223/rtiai/ai-demonstrator/prototype/TIE_german_evaluation/markuplm-large-finetuned-websrc",
                                                    cache_dir="cache")
            tokenizer = MarkupLMTokenizer.from_pretrained("/Users/jostgotte/Documents/Uni/WS2223/rtiai/ai-demonstrator/prototype/TIE_german_evaluation/markuplm-large-finetuned-websrc",
                                                        do_lower_case=False, cache_dir="cache")

            model = MarkupLMForQuestionAnswering.from_pretrained("/Users/jostgotte/Documents/Uni/WS2223/rtiai/ai-demonstrator/prototype/TIE_german_evaluation/markuplm-large-finetuned-websrc",
                                                                    from_tf=bool('.ckpt' in "/Users/jostgotte/Documents/Uni/WS2223/rtiai/ai-demonstrator/prototype/TIE_german_evaluation/markuplm-large-finetuned-websrc"),
                                                                    config=config, cache_dir="cache")
     
        return model, tokenizer

    def build_inputs(self,question, html):
        html_code = bs4.BeautifulSoup(html)
        
        raw_text_list, tag_num = html_to_text_list(html_code)
        page_text = ' '.join(raw_text_list)
        doc_tokens = []
        char_to_word_offset = []
        prev_is_whitespace = True
        for c in page_text:
            if is_whitespace(c):
                prev_is_whitespace = True
            else:
                if prev_is_whitespace:
                    doc_tokens.append(c)
                else:
                    doc_tokens[-1] += c
                prev_is_whitespace = False
            char_to_word_offset.append(len(doc_tokens) - 1)
        doc_tokens.append('no')
        char_to_word_offset.append(len(doc_tokens) - 1)
        doc_tokens.append('yes')
        char_to_word_offset.append(len(doc_tokens) - 1)

        
        tag_list = []
        tok_to_orig_index = []
        orig_to_tok_index = []
        all_doc_tokens = []
        for (i, token) in enumerate(doc_tokens):
            orig_to_tok_index.append(len(all_doc_tokens))
            if token in tag_list:
                sub_tokens = [token]
            else:
                sub_tokens = self.tokenizer.tokenize(token)
            for sub_token in sub_tokens:
                tok_to_orig_index.append(i)
                all_doc_tokens.append(sub_token)

        tok_to_tags_index, tags_to_tok_index, orig_tags, unique_tids = subtoken_tag_offset(
                                                                                            html_code,
                                                                                            tok_to_orig_index,
                                                                                            orig_to_tok_index)
        xpath_tag_map, xpath_subs_map = get_xpath_and_treeid4tokens(html_code, unique_tids,max_depth=50)

        assert tok_to_tags_index[-1] == tag_num - 1, (tok_to_tags_index[-1], tag_num - 1)

        self.example = TIEExample(
            doc_tokens=doc_tokens,
            qas_id='jo100000100000',
            html_tree=bs4.BeautifulSoup(html),
            question_text=question,
            orig_answer_text=None,
            answer_tid=None,
            start_position=None,
            end_position=None,
            tok_to_orig_index=tok_to_orig_index,
            orig_to_tok_index=orig_to_tok_index,
            all_doc_tokens=all_doc_tokens,
            tok_to_tags_index=tok_to_tags_index,
            tags_to_tok_index=tags_to_tok_index,
            orig_tags=orig_tags,
            tag_depth=calculate_depth(html_code),
            xpath_tag_map=xpath_tag_map,
            xpath_subs_map=xpath_subs_map,
        )        
        # convert to feature
        self.features = convert_examples_to_features(examples=[self.example],
                                                tokenizer=self.tokenizer,
                                                max_seq_length=384,
                                                doc_stride=128,
                                                max_query_length=64,
                                                max_tag_length=512,
                                                is_training=not True,
                                                cls_token=self.tokenizer.cls_token,
                                                sep_token=self.tokenizer.sep_token,
                                                pad_token=self.tokenizer.pad_token_id,)
        #build dict/inputs
        # Convert to Tensors and build dataset
        all_input_ids = torch.tensor([f.input_ids for f in self.features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in self.features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in self.features], dtype=torch.long)
        all_app_tags = [f.app_tags for f in self.features]
        all_example_index = [f.example_index for f in self.features]
        all_html_trees = [self.example.html_tree]
        all_base_index = [f.base_index for f in self.features]
        all_tag_to_token = [f.tag_to_token_index for f in self.features]
        all_page_id = [f.page_id for f in self.features]

        all_xpath_tags_seq = torch.tensor([f.xpath_tags_seq for f in self.features], dtype=torch.long)
        all_xpath_subs_seq = torch.tensor([f.xpath_subs_seq for f in self.features], dtype=torch.long)
    
    
        all_feature_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
        
        dataset = StrucDataset(all_input_ids, all_input_mask, all_segment_ids, all_feature_index,
                               all_xpath_tags_seq, all_xpath_subs_seq,
                               gat_mask=(all_app_tags, all_example_index, all_html_trees), base_index=all_base_index,
                               tag2tok=all_tag_to_token, shape=(512, 384),
                               training=False, page_id=all_page_id, mask_method=0,
                               mask_dir=os.path.dirname("data/dummy_file.json"), direction="B")
        


        eval_sampler = SequentialSampler(dataset) 
        self.eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=8)
        
    def write_predictions_provided_tag(self,all_examples, all_features, all_results, n_best_size, max_answer_length,
                                    do_lower_case, output_prediction_file, input_tag_prediction_file,
                                    output_refined_tag_prediction_file, output_nbest_file, verbose_logging,
                                    write_pred):
        r"""
        Providing the n best answer tag predictions, compute and write down the final answer span prediction results,
        including the n best results.

        Arguments:
            all_examples (list[SRCExample]): all the SRC Example of the dataset; note that we only need it to provide the
                                            mapping from example index to the question-answers id.
            all_features (list[InputFeatures]): all the features for the input doc spans.
            all_results (list[RawResult]): all the results from the models.
            n_best_size (int): the number of the n best buffer and the final n best result saved.
            max_answer_length (int): constrain the model to predict the answer no longer than it.
            do_lower_case (bool): whether the model distinguish upper and lower case of the letters.
            output_prediction_file (str): the file which the best answer text predictions will be written to.
            input_tag_prediction_file (str/dict): the file which the n best answer tag predictions has been written to, or
                                                the n best answer tag prediction results.
            output_refined_tag_prediction_file (str): the file which the refined best answer tag predictions will be written
                                                    to.
            output_nbest_file (str): the file which the n best answer predictions including text, tag, and probabilities
                                    will be written to.
            verbose_logging (bool): if true, all the warnings related to data processing will be printed.
            write_pred (bool): whether to write the predictions to the disk.
        Return:
            dict: the best answer span prediction results.
            dict: the refined best answer tag prediction results.
        """

        def _get_tag_id(ind2tok, start_ind, end_ind, base, ind2tag):
            tag_ind = -1
            for ind in range(base, len(ind2tok)):
                if (start_ind >= ind2tok[ind][0]) and (end_ind <= ind2tok[ind][1]):
                    tag_ind = ind
            tag_ind -= base
            assert tag_ind >= 0
            return ind2tag[tag_ind]

        example_index_to_features = collections.defaultdict(list)
        for feature in all_features:
            example_index_to_features[feature.example_index].append(feature)

        unique_id_to_result = {}
        for result in all_results:
            unique_id_to_result[result.unique_id] = result

        _PrelimPrediction = collections.namedtuple(  # pylint: disable=invalid-name
            "PrelimPrediction",
            ["feature_index",
            "tag_index", "start_index", "end_index",
            "tag_logit", "start_logit", "end_logit",
            "tag_id"])

        all_predictions = collections.OrderedDict()
        all_nbest_json = collections.OrderedDict()
        all_refined_tag_predictions = collections.OrderedDict()
        if isinstance(input_tag_prediction_file, str):
            all_tag_predictions = json.load(open(input_tag_prediction_file))
        else:
            all_tag_predictions = input_tag_prediction_file

        for (example_index, example) in enumerate(all_examples):
            features = example_index_to_features[example_index]
            nb_tag_pred = all_tag_predictions[example.qas_id]

            prelim_predictions = []
            for (feature_index, feature) in enumerate(features):
                result = unique_id_to_result[feature.unique_id]
                for item in nb_tag_pred:
                    tag_pred = item['tag_id']
                    if tag_pred not in feature.app_tags:
                        continue
                    tag_index = feature.app_tags.index(tag_pred) + feature.base_index
                    left_bound, right_bound = feature.tag_to_token_index[tag_index]
                    start_indexes = _get_best_indexes(result.start_logits[left_bound:right_bound + 1], n_best_size)
                    end_indexes = _get_best_indexes(result.end_logits[left_bound:right_bound + 1], n_best_size)
                    start_indexes = [ind + left_bound for ind in start_indexes]
                    end_indexes = [ind + left_bound for ind in end_indexes]
                    tag_logit = item['tag_logit']
                    for start_index in start_indexes:
                        for end_index in end_indexes:
                            # We could hypothetically create invalid predictions, e.g., predict
                            # that the start of the span is in the question. We throw out all
                            # invalid predictions.
                            if not feature.token_is_max_context.get(start_index, False):
                                continue
                            if end_index < start_index:
                                continue
                            length = end_index - start_index + 1
                            if length > max_answer_length:
                                continue
                            tag_ids = _get_tag_id(feature.tag_to_token_index,
                                                start_index, end_index,
                                                feature.base_index, feature.app_tags)
                            prelim_predictions.append(
                                _PrelimPrediction(
                                    feature_index=feature_index,
                                    tag_index=tag_index,
                                    start_index=start_index,
                                    end_index=end_index,
                                    tag_logit=tag_logit,
                                    start_logit=result.start_logits[start_index],
                                    end_logit=result.end_logits[end_index],
                                    tag_id=tag_ids))
            prelim_predictions = sorted(
                prelim_predictions,
                key=lambda x: (x.start_logit + x.end_logit),
                reverse=True)

            _NBestPrediction = collections.namedtuple(
                "NBestPrediction", ["text", "tag_logit", "start_logit", "end_logit", "tag_id"])

            seen_predictions = {}
            nbest = []
            for pred in prelim_predictions:
                if len(nbest) >= n_best_size:
                    break
                feature = features[pred.feature_index]
                tok_tokens = feature.tokens[pred.start_index:(pred.end_index + 1)]
                orig_doc_start = feature.token_to_orig_map[pred.start_index]
                orig_doc_end = feature.token_to_orig_map[pred.end_index]
                orig_tokens = example.doc_tokens[orig_doc_start:(orig_doc_end + 1)]
                tok_text = " ".join(tok_tokens)

                # De-tokenize WordPieces that have been split off.
                tok_text = tok_text.replace(" ##", "")
                tok_text = tok_text.replace("##", "")

                # Clean whitespace
                tok_text = tok_text.strip()
                tok_text = " ".join(tok_text.split())
                orig_text = " ".join(orig_tokens)

                final_text = get_final_text(tok_text, orig_text, do_lower_case, verbose_logging)
                if '{} with the tag_id of {}'.format(final_text, str(pred.tag_index)) in seen_predictions:
                    continue
                seen_predictions['{} with the tag_id of {}'.format(final_text, str(pred.tag_index))] = True

                nbest.append(
                    _NBestPrediction(
                        text=final_text,
                        tag_logit=pred.tag_logit,
                        start_logit=pred.start_logit,
                        end_logit=pred.end_logit,
                        tag_id=pred.tag_id))

            # In very rare edge cases we could have no valid predictions. So we
            # just create a nonce prediction in this case to avoid failure.
            if not nbest:
                nbest.append(
                    _NBestPrediction(
                        text='empty',
                        start_logit=0.0,
                        end_logit=0.0,
                        tag_logit=0.0,
                        tag_id=-1))

            assert len(nbest) >= 1

            total_scores = []
            for entry in nbest:
                total_scores.append(entry.start_logit + entry.end_logit)

            probs = _compute_softmax(total_scores)

            nbest_json = []
            for (i, entry) in enumerate(nbest):
                output = collections.OrderedDict()
                output["text"] = entry.text
                output["probability"] = probs[i]
                output["tag_logit"] = entry.tag_logit
                output["start_logit"] = entry.start_logit
                output["end_logit"] = entry.end_logit
                output["tag_id"] = entry.tag_id
                nbest_json.append(output)
            assert len(nbest_json) >= 1

            for idx, entry in enumerate(nbest_json):
                if entry["probability"] >= 0.8:
                    best_text = entry["text"].split()
                    best_text = ' '.join([w for w in best_text if w[0] != '<' or w[-1] != '>'])
                    all_predictions[idx] = best_text
                    best_tag = entry["tag_id"]
                    all_refined_tag_predictions[idx] = best_tag
            all_nbest_json[example.qas_id] = nbest_json

            if len(all_predictions) == 0:
                all_predictions[0] = "Es wurde keine Antwort gefunden"

        if write_pred:
            with open(output_prediction_file, "w") as writer:
                writer.write(json.dumps(all_predictions, indent=4) + "\n")

            with open(output_nbest_file, "w") as writer:
                writer.write(json.dumps(all_nbest_json, indent=4) + "\n")

            with open(output_refined_tag_prediction_file, 'w') as writer:
                writer.write(json.dumps(all_refined_tag_predictions, indent=4) + '\n')

        return all_predictions, all_refined_tag_predictions
        


    def predict(self ):
        all_results = []
        for batch in tqdm(self.eval_dataloader, desc="Evaluating"):
            self.model.eval()
            batch = tuple(t.to(torch.device("cpu")) for t in batch)
            with torch.no_grad():
                if self.stage == 2:
                    inputs = {'input_ids': batch[0],
                          'attention_mask': batch[1],
                          'token_type_ids': batch[2]}
                else:
                    inputs = {'input_ids'      : batch[0],
                            'attention_mask' : batch[1],
                            'token_type_ids' : batch[2],
                            'dom_mask'       : batch[-2],
                            'tag_to_tok'     : batch[-1]}
                    
                    inputs.update({'spa_mask': batch[-3]})
                inputs.update({
                    'xpath_tags_seq': batch[4],
                    'xpath_subs_seq': batch[5],
                })
                del inputs['token_type_ids']
                
                feature_indices = batch[3]
                outputs = self.model(**inputs)
        
            
            for i, feature_index in enumerate(feature_indices):
                eval_feature = self.features[feature_index.item()]
                unique_id = int(eval_feature.unique_id)
                
                    
                if self.stage == 2:
                    result = RawResult(unique_id=unique_id,
                                        start_logits=to_list(outputs[0][i]),
                                        end_logits=to_list(outputs[1][i]))
                else:
                    result = RawTagResult(unique_id=unique_id,
                                            tag_logits=to_list(outputs[0][i]))
                    
                all_results.append(result)

        # Compute predictions
        
        output_prediction_file = os.path.join(f"output/stage{self.stage}", "predictions_{}.json".format(""))
        output_tag_prediction_file = os.path.join(f"output/stage{self.stage}", "tag_predictions_{}.json".format(""))
        output_nbest_file = os.path.join(f"output/stage{self.stage}", "nbest_predictions_{}.json".format(""))
        output_result_file = os.path.join(f"output/stage{self.stage}", "qas_eval_results_{}.json".format(""))
        output_file = os.path.join(f"output/stage{self.stage}", "eval_matrix_results_{}".format(""))

        
        if self.stage==2:
            returns = self.write_predictions_provided_tag(all_examples=[self.example], all_features=self.features, all_results=all_results, n_best_size=20,
                                                max_answer_length=30, do_lower_case=False, output_prediction_file=output_prediction_file,
                                                input_tag_prediction_file="output/stage1/nbest_predictions_.json", output_refined_tag_prediction_file=output_tag_prediction_file,
                                                output_nbest_file=output_nbest_file, verbose_logging=False, write_pred=True)

                                                
        else:
            returns = write_tag_predictions([self.example], self.features, all_results, 20, "markuplm",
                                            output_tag_prediction_file, output_nbest_file, write_pred=True)
            output_prediction_file = None
            

        return returns

         
    
    def answer_question(self, question, html):
        
        self.stage = 1
        
        intermediate_answer = self.predict()

        self.stage = 2

        self.model, self.tokenizer = self.build_model_and_tokenizer()

        final_answer = self.predict()

        return final_answer
    
    def monitor(self, url, question):
        html = write_html(fetch_html(url))
        self.build_inputs(question, html)
        answers = self.answer_question(question=question,  html=html)
        for key in answers[0]: st.write(answers[0][key])
        while True:
            time.sleep(300)
            new_html = fetch_html(url)
            if new_html == read_html():
                st.write('Nothing has changed')
            else:
                st.write('Something has changed')
                write_html(new_html)
                st.write(self.answer_question( html=html, question=question))


def state_change():
    print("1")
    with open("page/temp.html", "w") as write, open("page/HPI Connect Jobportal.html", "r") as read:
        write.write(read.read())
    print("2")
    with open("page/HPI Connect Jobportal.html", "w") as write, open("page/HPI Connect Jobportal1.html", "r") as read:
        write.write(read.read())
    print("3")
    with open("page/HPI Connect Jobportal1.html", "w") as write, open("page/temp.html", "r") as read:
        write.write(read.read())
   

