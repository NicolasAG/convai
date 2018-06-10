"""
This script previously extracted test amt data and label the chosen message
in each conversation interaction according to some specific short term
ranker, previously trained for the ConvAI competition.
"""
import logging
import numpy as np
import cPickle as pkl
import re
import spacy
import json

import torch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


logging.basicConfig(level=logging.DEBUG, format="%(asctime)s: %(name)s: %(levelname)s: %(message)s")

logger = logging.getLogger(__name__)

nlp = spacy.load('en')

config = {
    "wh_words": ['who', 'where', 'when', 'why', 'what', 'how', 'whos', 'wheres', 'whens', 'whys', 'whats', 'hows']
}

# list of generic words to detect if user is bored
generic_words_list = []
with open('../data/generic_list.txt') as fp:
    for line in fp:
        generic_words_list.append(line.strip())
generic_words_list = set(generic_words_list)  # remove duplicates


def get_test_data():
    """
    Load raw data. Takes a while!
    """

    # regular
    data_f = "./data/q_ranker_amt_data_1527760664.38.pkl"

    logger.info("")
    logger.info("Loading data from %s..." % data_f)
    with open(data_f, 'rb') as f:
        raw_data = pkl.load(f)

    test_data = raw_data['test'][0]
    test_size = raw_data['test'][1]
    logger.info("got %d test examples" % test_size)

    return test_data


def regroup_chats(raw_data):
    """
    Build a map from chat ID to list of (context, candidates, rewards, predictions) list
    :return: a map of this form:
    chat_id : [
        {
            'article'      : string,
            'context'      : [list of strings],
            'candidates'   : [list of strings],
            'model_names'  : [list of strings],
            'ranker_confs' : [list of floats],
            'ranker_scores': [list of floats],
            'rewards'      : [list of ints],
            'predictions_oldRules' : [list of floats]
            'predictions_oldClass' : [list of floats]
        },
        {
            ...
        },
        ...
    ],
    chat_id : [
        ...
    ],
    """
    logger.info("")
    logger.info("Re-arranging data into list of (article, context, list of candidates)...")

    chats = {}

    for article, entries in raw_data.iteritems():
        for entry in entries:
            '''
            'chat_id': <string>,
            'state': <list of strings> ie: context,
            'action': {
                'candidate': <string>  ie: candidate,
                'custom_enc': <list of float>,
                'model_name': <string of model name>,
                'conf': <string of float between 0.0 & 1.0 or int 0 when not evaluated>,
                'score': <string of float between 0.0 & 5.0 or int -1 when not evaluated>,
            },
            'reward': <int {0,1}>,
            'next_state': <list of strings || None> ie: next_context,
            'next_actions': <list of actions || None> ie: next possible actions
            'quality': <int {1,2,3,4,5}>,
            '''
            # if chat already exists,
            if entry['chat_id'] in chats:
                idx = -1
                for i, c in enumerate(chats[entry['chat_id']]):
                    # if context already exists, add this candidate response
                    if c['context'] == entry['state']:
                        assert c['article'] is not None
                        c['candidates'].append(entry['action']['candidate'])
                        c['model_names'].append(entry['action']['model_name'])
                        c['ranker_confs'].append(entry['action']['conf'])
                        c['ranker_scores'].append(entry['action']['score'])
                        c['rewards'].append(entry['reward'])
                        c['predictions_oldRules'].append(0.0)  # placeholder for now
                        c['predictions_oldClass'].append(0.0)  # placeholder for now
                        idx = i
                        break
                # if context doesn't exists, add it as a new one with this candidate response
                if idx == -1:
                    chats[entry['chat_id']].append({
                        'article': article,
                        'context': entry['state'],
                        'candidates': [entry['action']['candidate']],
                        'model_names': [entry['action']['model_name']],
                        'ranker_confs': [entry['action']['conf']],
                        'ranker_scores': [entry['action']['score']],
                        'rewards': [entry['reward']],
                        'predictions_oldRules': [0.0],  # placeholder for now
                        'predictions_oldClass': [0.0]  # placeholder for now
                    })
            # if chat doesn't exists, add a new one
            else:
                chats[entry['chat_id']] = [{
                    'article': article,
                    'context': entry['state'],
                    'candidates': [entry['action']['candidate']],
                    'model_names': [entry['action']['model_name']],
                    'ranker_confs': [entry['action']['conf']],
                    'ranker_scores': [entry['action']['score']],
                    'rewards': [entry['reward']],
                    'predictions_oldRules': [0.0],  # placeholder for now
                    'predictions_oldClass': [0.0]  # placeholder for now
                }]

    logger.info("got %d chats." % len(chats))
    return chats


def test_individually(chats):
    logger.info("")
    logger.info("Now simulating the old chatbot decision policy...")

    def topic_match(text):
        # catch responses of the style "what is this article about"
        question_match_1 = ".*what\\s*(is|'?s|does)?\\s?(this|it|the)?\\s?(article)?\\s?(talks?)?\\s?(about)\\s*(\\?)*"
        # catch also responses of the style : "what do you think of this article"
        question_match_2 = ".*what\\sdo\\syou\\sthink\\s(of|about)\\s(this|it|the)?\\s?(article)\\s*\\?*"
        return re.match(question_match_1, text, re.IGNORECASE) or re.match(question_match_2, text, re.IGNORECASE)

    def has_wh_words(nt_words):
        # check if the text contains wh words
        for word in nt_words:
            if word in set(config['wh_words']):
                return True
        return False

    def rank_candidate_responses(c):
        # array containing tuple of (model_name, rank_score)
        consider_models = []
        dont_consider_models = []

        always_consider = ['hred-reddit', 'hred-twitter', 'alicebot']

        for idx, response in enumerate(c['candidates']):
            model = c['model_names'][idx]
            conf = float(c['ranker_confs'][idx])
            score = float(c['ranker_scores'][idx])

            if conf > 0.75 and model in always_consider:
                consider_models.append((model, conf * score))

            if conf < 0.25 and conf != 0.0 and model != 'fact_gen':
                # keep fact generator model for failure handling case
                dont_consider_models.append((model, conf * score))

        consider = None
        dont_consider = None
        if len(consider_models) > 0:
            consider_models = sorted(consider_models, key=lambda x: x[1], reverse=True)
            consider = consider_models[0][0]

        elif len(dont_consider_models) > 0:
            dont_consider = dont_consider_models

        return consider, dont_consider

    def was_it_bored(context):
        bored_cnt = 0
        for utt in context:
            # nouns of the user message
            ntext = nlp(unicode(utt))
            nt_words = [p.lemma_ for p in ntext]

            # check if user said only generic words:
            generic_turn = True
            for word in nt_words:
                if word not in generic_words_list:
                    generic_turn = False
                    break

            # if text contains 2 words or less, add 1 to the bored count
            # also consider the case when the user says only generic things
            if len(utt.strip().split()) <= 2 or generic_turn:
                bored_cnt += 1

        return bored_cnt > 0 and bored_cnt % 2 == 0

    for chat_id, list_of_contexts in chats.iteritems():
        for c in list_of_contexts:
            ''' {
                'article'      : string,
                'context'      : [list of strings],
                'candidates'   : [list of strings],
                'model_names'  : [list of strings],
                'ranker_confs' : [list of string numbers],
                'ranker_scores': [list of string numbers],
                'rewards'      : [list of ints],
                'predictions_oldRules': [list of floats],
                'predictions_oldClass': [list of floats]
            } '''
            assert len(c['candidates']) == len(c['model_names']) == len(c['ranker_confs']) == len(c['ranker_scores'])\
                   == len(c['rewards']) == len(c['predictions_oldRules']) == len(c['predictions_oldClass'])

            # for the noRule/onlyClassifier policy, put in the confidence scores as float values
            c['predictions_oldClass'] = map(lambda s: float(s), c['ranker_confs'])

            # step1: if context length = 1, chose randomly between NQG and EntitySentence
            if len(c['context']) == 1:
                assert len(c['candidates']) == 2, "number of candidates (%d) != 2" % len(c['candidates'])
                assert c['model_names'][0] in ['candidate_question', 'nqg'], "model name (%s) unknown" % c['model_names'][0]
                assert c['model_names'][1] in ['candidate_question', 'nqg'], "model name (%s) unknown" % c['model_names'][1]

                choice = np.random.choice([0, 1])
                c['predictions_oldRules'][choice] = 1.0
                continue  # go to next interaction

            # step2: if query falls under dumb questions, respond appropriately
            # if candidate responses contains dumb_qa then there must have been a match
            if 'dumb_qa' in c['model_names']:
                idx = c['model_names'].index('dumb_qa')
                assert c['ranker_confs'][idx] == '0', "ranker conf (%s) != '0'" % c['ranker_confs'][idx]
                assert c['ranker_scores'][idx] == '-1', "ranker score (%s) != '-1'" % c['ranker_scores'][idx]
                assert len(c['candidates']) == 9, "number of candidates (%d) != 9" % len(c['candidates'])

                c['predictions_oldRules'][idx] = 1.0
                continue  # go to next interaction

            # step3: if query falls under topic request, respond with the article topic
            if topic_match(c['context'][-1]) and 'topic_model' in c['model_names']:
                idx = c['model_names'].index('topic_model')
                assert c['ranker_confs'][idx] == '0', "ranker confs (%s) != '0'" % c['ranker_confs'][idx]
                assert c['ranker_scores'][idx] == '-1', "ranker scores (%s) != '-1'" % c['ranker_scores'][idx]

                c['predictions_oldRules'][idx] = 1.0
                continue  # go to next interaction

            # nouns of the previous user message
            ntext = nlp(unicode(c['context'][-1]))
            nt_words = [p.lemma_ for p in ntext]

            # step4: if query is a question, try to reply with DrQA
            if has_wh_words(nt_words) or ('which' in c['context'][-1] and '?' in c['context'][-1]):
                # nouns from the article
                article_nlp = nlp(unicode(c['article']))
                article_nouns = [p.lemma_ for p in article_nlp if p.pos_ in ['NOUN', 'PROPN']]

                common = list(set(article_nouns).intersection(set(nt_words)))

                # if there is a common noun between question and article select DrQA
                if len(common) > 0 and 'drqa' in c['model_names']:
                    idx = c['model_names'].index('drqa')
                    c['predictions_oldRules'][idx] = 1.0
                    continue  # go to next interaction

            ###
            # Ranker based selection
            ###

            # get ignored models
            idx_to_ignore = []
            best_model, dont_consider = rank_candidate_responses(c)
            if dont_consider and len(dont_consider) > 0:
                for model, _ in dont_consider:
                    idx_to_ignore.append(c['model_names'].index(model))

            # Reduce confidence of CAND_QA
            if 'candidate_question' in c['model_names']:
                idx = c['model_names'].index('candidate_question')
                c['ranker_confs'][idx] = str(float(c['ranker_confs'][idx]) / 2)

            # step5: Bored model selection
            if was_it_bored(c['context']):
                # list of available models to use if bored
                bored_models = ['nqg', 'fact_gen', 'candidate_question']
                bored_idx = [c['model_names'].index(m) for m in c['model_names'] if m in bored_models]
                bored_idx = [idx for idx in bored_idx if idx not in idx_to_ignore]
                # if user is bored, change the topic by asking a question
                if len(bored_idx) > 0:
                    # assign model selection probability based on estimator confidence
                    for idx in bored_idx:
                        c['predictions_oldRules'][idx] = float(c['ranker_confs'][idx])

                    c['predictions_oldRules'] = c['predictions_oldRules'] / np.sum(c['predictions_oldRules'])
                    c['predictions_oldRules'] = c['predictions_oldRules'].tolist()
                    continue  # go to next interaction

            # step6: If not bored, then select from best model
            if best_model:
                idx = c['model_names'].index(best_model)
                c['predictions_oldRules'][idx] = 1.0
                continue  # go to next interaction

            # step7: Sample from the other models based on confidence probability
            # randomly decide a model to query to get a response:
            models = ['hred-reddit', 'hred-twitter', 'alicebot']
            # if the user asked a question, also consider DrQA
            if has_wh_words(nt_words) or ('which' in c['context'][-1] and '?' in c['context'][-1]):
                models.append('drqa')
            # if the user didn't ask a question, also consider models that ask questions: nqg, and cand_qa
            else:
                models.extend(['nqg', 'candidate_question'])

            available_idx = [c['model_names'].index(m) for m in c['model_names'] if m in models]
            available_idx = [idx for idx in available_idx if idx not in idx_to_ignore]
            if len(available_idx) > 0:
                # assign model selection probability based on estimator confidence
                for idx in available_idx:
                    c['predictions_oldRules'][idx] = float(c['ranker_confs'][idx])

                c['predictions_oldRules'] = c['predictions_oldRules'] / np.sum(c['predictions_oldRules'])
                c['predictions_oldRules'] = c['predictions_oldRules'].tolist()
                continue  # go to next interaction

            # last step: if still no response, then just send a random fact
            if 'fact_gen' in c['model_names']:
                idx = c['model_names'].index('fact_gen')
                c['predictions_oldRules'][idx] = 1.0
                continue  # go to next interaction

            # at this stage only one response should have been selected!
            logger.warning("THIS SHOULD NEVER BE PRINTED... :3")

    return chats


def get_recall(chats, policy):
    assert policy in ['RuleBased', 'Sample', 'Argmax']
    recalls = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    total = 0.0

    for chat_id, contexts in chats.iteritems():
        for c in contexts:
            '''
            {
                'article'      : string,
                'context'      : [list of strings],
                'candidates'   : [list of strings],
                'model_names'  : [list of strings],
                'ranker_confs' : [list of string numbers],
                'ranker_scores': [list of string numbers],
                'rewards'      : [list of ints],
                'predictions_oldRules': [list of floats],
                'predictions_oldClass': [list of floats]
            },
            '''
            # sum of all candidate rewards is at most 1
            assert np.sum(c['rewards']) <= 1

            # sum of all predictions with ruleBased policy is exactly 1
            assert np.isclose(np.sum(c['predictions_oldRules']), 1.0)

            # Get sorted idx of candidate scores
            if policy == 'RuleBased':
                # select response according to rule based
                _, sorted_idx = torch.Tensor(c['predictions_oldRules']).sort(descending=True)
            elif policy == 'Argmax':
                # select the max response
                _, sorted_idx = torch.Tensor(c['predictions_oldClass']).sort(descending=True)
            elif policy == 'Sample':
                # sample response according to probability score
                min_value = min([p for p in c['predictions_oldClass'] if p > 0])
                min_value /= 10.
                #print c['predictions_oldClass'], min_value
                probas = map(lambda p: min_value if p == 0.0 else p, c['predictions_oldClass'])
                #print probas
                probas = probas / np.sum(probas)
                #print probas
                sorted_idx = np.random.choice(range(len(probas)),
                                              size=len(probas),
                                              replace=False,
                                              p=probas)
            else:
                raise ValueError("Unkown policy: %s" % policy)

            # idx of chosen candidate
            idx = np.argmax(c['rewards'])

            # Recall @ 1
            if idx in sorted_idx[:1]:
                for i in range(9):
                    recalls[i] += 1
            # Recal @ 2
            elif idx in sorted_idx[:2]:
                for i in range(1, 9):
                    recalls[i] += 1
            # Recal @ 3
            elif idx in sorted_idx[:3]:
                for i in range(2, 9):
                    recalls[i] += 1
            # Recal @ 4
            elif idx in sorted_idx[:4]:
                for i in range(3, 9):
                    recalls[i] += 1
            # Recal @ 5
            elif idx in sorted_idx[:5]:
                for i in range(4, 9):
                    recalls[i] += 1
            # Recal @ 6
            elif idx in sorted_idx[:6]:
                for i in range(5, 9):
                    recalls[i] += 1
            # Recal @ 7
            elif idx in sorted_idx[:7]:
                for i in range(6, 9):
                    recalls[i] += 1
            # Recal @ 8
            elif idx in sorted_idx[:8]:
                for i in range(7, 9):
                    recalls[i] += 1
            # Recal @ 9
            elif idx in sorted_idx[:9]:
                for i in range(8, 9):
                    recalls[i] += 1

            total += 1

    return recalls, total


def recallat1_contextlen(chats, policy):
    assert policy in ['RuleBased', 'Sample', 'Argmax']
    recalls = [0.0, 0.0, 0.0, 0.0, 0.0,
               0.0, 0.0, 0.0, 0.0, 0.0,
               0.0, 0.0, 0.0, 0.0, 0.0,
               0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    totals = [0.0, 0.0, 0.0, 0.0, 0.0,
               0.0, 0.0, 0.0, 0.0, 0.0,
               0.0, 0.0, 0.0, 0.0, 0.0,
               0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    for chat_id, contexts in chats.iteritems():
        for c in contexts:
            '''
            {
                'article'      : string,
                'context'      : [list of strings],
                'candidates'   : [list of strings],
                'model_names'  : [list of strings],
                'ranker_confs' : [list of string numbers],
                'ranker_scores': [list of string numbers],
                'rewards'      : [list of ints],
                'predictions_oldRules': [list of floats],
                'predictions_oldClass': [list of floats]
            },
            '''
            # Get sorted idx of candidate scores
            if policy == 'RuleBased':
                # select response according to rule based
                _, sorted_idx = torch.Tensor(c['predictions_oldRules']).sort(descending=True)
            elif policy == 'Argmax':
                # select the max response
                _, sorted_idx = torch.Tensor(c['predictions_oldClass']).sort(descending=True)
            elif policy == 'Sample':
                # sample response according to probability score
                min_value = min([p for p in c['predictions_oldClass'] if p > 0])
                min_value /= 10.
                #print c['predictions_oldClass']
                probas = map(lambda p: min_value if p == 0.0 else p, c['predictions_oldClass'])
                #print probas
                probas = probas / np.sum(probas)
                # print probas
                sorted_idx = np.random.choice(range(len(probas)),
                                              size=len(probas),
                                              replace=False,
                                              p=probas)
            else:
                raise ValueError("Unkown policy: %s" % policy)

            # idx of chosen candidate
            idx = np.argmax(c['rewards'])

            # Recall @ 1
            if idx in sorted_idx[:1]:
                try:
                    recalls[len(c['context']) - 1] += 1.
                except IndexError:
                    recalls[-1] += 1.

            try:
                totals[len(c['context']) - 1] += 1.
            except IndexError:
                totals[-1] += 1.

    return recalls, totals


def main():

    # Load AMT test data
    raw_data = get_test_data()

    # Build report
    chats = regroup_chats(raw_data)

    # predict chosen candidate
    chats = test_individually(chats)

    ###
    # Saving report
    ###
    logger.info("")
    logger.info("Saving report...")
    with open("./models/short_term_on_amt_report.json", 'wb') as f:
        json.dump(chats, f, indent=2)
    logger.info("done.")

    ###
    # Measuring accuracy
    ###
    barwidth = 0.3

    logger.info("")
    logger.info("Measuring recall at predicting best candidate...")

    # Get recall for different selection strategies
    recalls_rulebased, total_rulebased = get_recall(chats, 'RuleBased')
    recalls_argmax, total_argmax = get_recall(chats, 'Argmax')
    recalls_sample, total_sample = get_recall(chats, 'Sample')

    logger.info("Predicted like human behavior with rulebased selection:")
    for idx, r in enumerate(recalls_rulebased):
        logger.info("- recall@%d: %d / %d = %g" % (
            idx + 1, r, total_rulebased, r / total_rulebased
        ))
    logger.info("Predicted like human behavior with argmax selection:")
    for idx, r in enumerate(recalls_argmax):
        logger.info("- recall@%d: %d / %d = %g" % (
            idx + 1, r, total_argmax, r / total_argmax
        ))
    logger.info("Predicted like human behavior with sampled selection:")
    for idx, r in enumerate(recalls_sample):
        logger.info("- recall@%d: %d / %d = %g" % (
            idx + 1, r, total_sample, r / total_sample
        ))

    # - plot recalls: r@1, r@2, ..., r@9
    recalls_rulebased = np.array(recalls_rulebased) / total_rulebased
    recalls_argmax = np.array(recalls_argmax) / total_argmax
    recalls_sample = np.array(recalls_sample) / total_sample
    plt.bar(range(len(recalls_rulebased)),
            recalls_rulebased,
            color='r', width=barwidth,
            label='Rule-Based')
    plt.bar([x + barwidth for x in range(len(recalls_argmax))],
            recalls_argmax,
            color='b', width=barwidth,
            label='Argmax')
    plt.bar([x + 2*barwidth for x in range(len(recalls_sample))],
            recalls_sample,
            color='g', width=barwidth,
            label='Sampled')
    plt.legend(loc='best')
    plt.title("Recall@k measure")
    plt.xlabel("k")
    plt.xticks(
        [r + barwidth for r in range(len(recalls_rulebased))],
        range(1, len(recalls_rulebased) + 1)
    )
    plt.ylabel("recall")
    plt.savefig("./models/short_term_on_amt_recalls.png")
    plt.close()

    ###
    # Compute recall@1 for each context length
    ###
    logger.info("")
    logger.info("Measuring recall@1 for each context length...")

    # Get recall for different selection strategies
    recalls_rulebased, totals_rulebased = recallat1_contextlen(chats, 'RuleBased')
    recalls_argmax, totals_argmax = recallat1_contextlen(chats, 'Argmax')
    recalls_sample, totals_sample = recallat1_contextlen(chats, 'Sample')

    logger.info("Predicted like human behavior with rulebased selection:")
    for c_len, r in enumerate(recalls_rulebased):
        logger.info("- recall@1 for context of size %d: %d / %d = %s" % (
            c_len + 1, r, totals_rulebased[c_len], (r / totals_rulebased[c_len] if totals_rulebased[c_len] > 0 else "inf")
        ))
    logger.info("Predicted like human behavior with argmax selection:")
    for c_len, r in enumerate(recalls_argmax):
        logger.info("- recall@1 for context of size %d: %d / %d = %s" % (
            c_len + 1, r, totals_argmax[c_len], (r / totals_argmax[c_len] if totals_argmax[c_len] > 0 else "inf")
        ))
    logger.info("Predicted like human behavior with sampled selection:")
    for c_len, r in enumerate(recalls_sample):
        logger.info("- recall@1 for context of size %d: %d / %d = %s" % (
            c_len + 1, r, totals_sample[c_len], (r / totals_sample[c_len] if totals_sample[c_len] > 0 else "inf")
        ))

    # - plot recalls:
    recalls_rulebased = np.array(recalls_rulebased) / np.array(totals_rulebased)
    recalls_argmax = np.array(recalls_argmax) / np.array(totals_argmax)
    recalls_sample = np.array(recalls_sample) / np.array(totals_sample)
    plt.bar(range(len(recalls_rulebased)),
            recalls_rulebased,
            color='r', width=barwidth,
            label='Rule-Based')
    plt.bar([x + barwidth for x in range(len(recalls_argmax))],
            recalls_argmax,
            color='b', width=barwidth,
            label='Argmax')
    plt.bar([x + 2 * barwidth for x in range(len(recalls_sample))],
            recalls_sample,
            color='g', width=barwidth,
            label='Sampled')
    plt.legend(loc='best')
    plt.title("Recall@1 for each context length")
    plt.xlabel("#of messages")
    plt.xticks(
        [r + barwidth for r in range(len(recalls_rulebased))],
        range(1, len(recalls_rulebased)) + ['>']
    )
    plt.ylabel("recall")
    plt.savefig("./models/short_term_on_amt_recall1_c-len.png")
    plt.close()

    logger.info("done.")


if __name__ == '__main__':
    main()

