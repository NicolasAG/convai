import numpy as np
import pymongo
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DB_PORT = 8091
DB_CLIENT = '132.206.3.23'


def get_convo_qualities(convos):
    n_quality = [0, 0, 0, 0, 0]  # number of conversations for each quality score: from 1 to 5

    for conv in convos:
        # count conversation quality:
        conv_quality = int(conv['chat_state']['metrics']['quality'])
        n_quality[conv_quality-1] += 1

    return n_quality


def get_model_usage(convos):
    """
    Collect messages from db.dialogs
    Build a mapping from generative model to counters
    :param convos: list of accepted conversation
    :return: the mapping
    """

    # map from model_name to number of time available & number of time used & number of time not available
    data = {
        'hred_twt': {'unavailable': 0., 'available': 0., 'used': 0.},
        'hred_red': {'unavailable': 0., 'available': 0., 'used': 0.},
        'nqg': {'unavailable': 0., 'available': 0., 'used': 0.},

        'drqa': {'unavailable': 0., 'available': 0., 'used': 0.},
        'topic': {'unavailable': 0., 'available': 0., 'used': 0.},
        'fact': {'unavailable': 0., 'available': 0., 'used': 0.},

        'entity_sent': {'unavailable': 0., 'available': 0., 'used': 0.},
        'simple_qst': {'unavailable': 0., 'available': 0., 'used': 0.},
        'alice': {'unavailable': 0., 'available': 0., 'used': 0.}
    }
    n_ex = 0

    for conv_idx, conv in enumerate(convos):
        # loop through each turn for this conversation
        for turn in conv['chat_state']['turns']:

            choice_idx = turn['choice']  # choice index
            # skip turn if invalid choice
            if choice_idx == '-1' or choice_idx == -1:
                continue

            # if chose to finish, this is the last turn so we can break
            if choice_idx == '/end':
                break

            # check if each model was a candidate response
            all_models = {'hred_twt': False, 'hred_red': False, 'nqg': False,
                          'drqa': False, 'topic': False, 'fact': False,
                          'entity_sent': False, 'simple_qst': False, 'alice': False}

            # loop through each candidate option
            for option_idx, option in turn['options'].items():
                '''
                "text" : <str>,
                "chat_unique_id" : <str>,
                "vote" : <str of either 1 or 0>,
                "score" : <str of float between 1 and 5>,
                "model_name" : <str>,
                "context": <list of strings>,
                "chat_id" : <str>,
				"conf" : <str of float between 0 and 1>
                '''
                # don't consider the /end option
                if option_idx == '/end':
                    continue

                # reset some model names (all except drqa & nqg)
                model_name = option['model_name']
                if model_name == "candidate_question":
                    model_name = "entity_sent"
                elif model_name == "dumb_qa":
                    model_name = "simple_qst"
                elif model_name == "fact_gen":
                    model_name = "fact"
                elif model_name == "topic_model":
                    model_name = "topic"
                elif model_name == "hred-reddit":
                    model_name = "hred_red"
                elif model_name == "hred-twitter":
                    model_name = "hred_twt"
                elif model_name == "alicebot":
                    model_name = "alice"

                # always add one when available
                data[model_name]['available'] += 1

                # flag as available
                all_models[model_name] = True

                # if chosen, add one to 'used' count
                if choice_idx == option_idx:
                    data[model_name]['used'] += 1

                # increment example counter
                n_ex += 1

            # end of options

            # count models that are not available
            for model_name in all_models:
                if not all_models[model_name]:
                    data[model_name]['unavailable'] += 1


        # end of conversation
    # end of accepted HITs
    return data, n_ex


def get_proportions(convos):
    candidate_counts = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    total_counts = 0.0

    context_lengths = [0.0, 0.0, 0.0, 0.0, 0.0,
                       0.0, 0.0, 0.0, 0.0, 0.0,
                       0.0, 0.0, 0.0, 0.0, 0.0,
                       0.0, 0.0, 0.0, 0.0, 0.0]
    total_lengths = 0.0
    long_contexts = 0.0

    total = 0.0

    missing_at_7 = {  # ignoring 'dumb_qa'
        'hred-twitter': 0., 'hred-reddit': 0., 'nqg': 0.,
        'drqa': 0., 'topic_model': 0., 'fact_gen': 0.,
        'candidate_question': 0., 'alicebot': 0.,
    }

    for conv_idx, conv in enumerate(convos):
        # loop through each turn for this conversation
        for turn in conv['chat_state']['turns']:
            '''
            "choice" : <str idx of chosen candidate>,
			"options" : { <str idx> : <option>, ... }
            '''

            choice_idx = turn['choice']  # choice index
            # skip turn if invalid choice
            if choice_idx == '-1' or choice_idx == -1:
                continue

            # if chose to finish, this is the last turn so we can break
            #if choice_idx == '/end':
            #    break

            c_len = 0  # current guess of the context length
            got_end_option = False  # to know when to ignore a candidate when measuring length

            # loop through each candidate option
            for option_idx, option in turn['options'].items():
                '''
                "text" : <str>,
                "chat_unique_id" : <str>,
                "vote" : <str of either 1 or 0>,
                "score" : <str of float between 1 and 5>,
                "model_name" : <str>,
                "context": <list of strings>,
                "chat_id" : <str>,
                "conf" : <str of float between 0 and 1>
                '''
                # don't consider the /end option
                if option_idx == '/end':
                    got_end_option = True
                    continue

                # update the context length
                if c_len == 0:
                    if len(turn['options']) > 2:
                        model_name = option['model_name']
                        c_len = len(option['context'])

                        if model_name == "hred-reddit":
                            c_len -= 1  # remove 1 to context length since the context also considers the candidate we return
                        elif model_name == "hred-twitter":
                            c_len -= 1  # remove 1 to context length since the context also considers the candidate we return
                        elif model_name == "nqg":
                            c_len = len(option['context'])

                        elif model_name == "drqa":
                            c_len -= 1  # remove 1 to context length since the context also considers the candidate we return
                        elif model_name == "topic_model":
                            c_len -= 1  # remove 1 to context length since the context also considers the candidate we return
                        elif model_name == "fact_gen":
                            c_len -= 1  # remove 1 to context length since the context also considers the candidate we return

                        elif model_name == "candidate_question":
                            c_len -= 1  # remove 1 to context length since the context also considers the candidate we return
                        elif model_name == "alicebot":
                            c_len -= 1  # remove 1 to context length since the context also considers the candidate we return
                    else:
                        c_len = 1

            if got_end_option:
                # -1 number of candidates because we don't consider the 'end' option
                candidate_counts[len(turn['options']) - 2] += 1
                total_counts += (len(turn['options']) - 1)
            else:
                candidate_counts[len(turn['options']) - 1] += 1
                total_counts += len(turn['options'])

                # DEBUG
                # print model names that are missing (except 'dumb_qa') when only 7 candidate responses are available
                if len(turn['options']) == 7:
                #if len(turn['options']) == 6:
                #if len(turn['options']) == 5:
                    all_names = set([
                        'hred-twitter', 'hred-reddit', 'nqg',
                        'drqa', 'topic_model', 'fact_gen',
                        'candidate_question', 'alicebot'])
                    names = set([op['model_name'] for op in turn['options'].values()])
                    missing = list(all_names - names)
                    assert len(missing) == 1
                    #if missing[0] == 'alicebot':
                    #    print "alice is missing... context below:"
                    #    print turn['options']['1']['context'][-3]
                    #    print turn['options']['1']['context'][-2]
                    #    print turn['options']['1']['context'][-1]
                    #    print ""
                    #elif missing[0] == 'topic_model':
                    #    print turn['options']['1']['context']
                    missing_at_7[missing[0]] += 1
                    #missing_at_7[missing[1]] += 1
                    #missing_at_7[missing[2]] += 1

            try:
                context_lengths[c_len-1] += 1
            except IndexError:
                long_contexts += 1
            total_lengths += c_len

            total += 1

    return (candidate_counts, total_counts), (context_lengths, long_contexts, total_lengths), total, missing_at_7



def main():
    # connect to database
    print "connect to database & query approved conversations..."
    client = pymongo.MongoClient(DB_CLIENT, DB_PORT)
    db = client.rllchat_mturk

    # get all accepted conversations
    results = list(db.dialogs.find({'review': 'accepted'}))
    print "got %d conversations." % len(results)

    ###
    # get quality of all conversations
    ###
    qualities = get_convo_qualities(results)
    n_conv = np.sum(qualities)
    assert n_conv == len(results)
    print " - 'very bad'  conversations: %d / %d = %f" % (qualities[0], n_conv, qualities[0] / float(n_conv))
    print " - 'bad'       conversations: %d / %d = %f" % (qualities[1], n_conv, qualities[1] / float(n_conv))
    print " - 'medium'    conversations: %d / %d = %f" % (qualities[2], n_conv, qualities[2] / float(n_conv))
    print " - 'good'      conversations: %d / %d = %f" % (qualities[3], n_conv, qualities[3] / float(n_conv))
    print " - 'very good' conversations: %d / %d = %f" % (qualities[4], n_conv, qualities[4] / float(n_conv))
    print ""

    # plot qualities
    qualities = np.array(qualities) / float(n_conv)
    plt.bar(range(len(qualities)), qualities, tick_label=['v.bad', 'bad', 'medium', 'good', 'v.good'])
    plt.xticks(fontsize=10, rotation=-30)
    plt.title("Conversation qualities")
    plt.xlabel("Label")
    plt.ylabel("Proportion")
    plt.savefig("amt_data_qualities.png")
    plt.close()

    ###
    # get model availabilities
    ###
    model_usage, n_ex = get_model_usage(results)
    for model_name, usage in model_usage.items():
        print " - %s\tavailability: %d / %d = %f" % (model_name, usage['available'], n_ex, usage['available'] / n_ex)
    print ""

    # plot availabilities
    model_names = model_usage.keys()
    availabilities = [usage['available'] / n_ex for usage in model_usage.values()]
    plt.bar(range(len(availabilities)), availabilities, tick_label=model_names)
    plt.xticks(fontsize=10, rotation=-30)
    plt.title("Proportion of available models")
    plt.xlabel("Model")
    plt.ylabel("Proportion")
    plt.savefig("amt_data_availabilities.png")
    plt.close()

    # plot UNavailabilities
    model_names = model_usage.keys()
    unavailabilities = [usage['unavailable'] / n_ex for usage in model_usage.values()]
    plt.bar(range(len(unavailabilities)), unavailabilities, tick_label=model_names)
    plt.xticks(fontsize=10, rotation=-30)
    plt.title("Proportion of unavailable models")
    plt.xlabel("Model")
    plt.ylabel("Proportion")
    plt.savefig("amt_data_unavailabilities.png")
    plt.close()

    # get model usage
    for model_name, usage in model_usage.items():
        print " - %s\tusage: %d / %d = %f" % (
            model_name, usage['used'], usage['available'], usage['used'] / usage['available']
        )

    # plot utilities
    model_names = model_usage.keys()
    utilities = [usage['used'] / usage['available'] for usage in model_usage.values()]
    plt.bar(range(len(utilities)), utilities, tick_label=model_names)
    plt.xticks(fontsize=10, rotation=-30)
    plt.title("Proportion of selected models when available")
    plt.xlabel("Model")
    plt.ylabel("Proportion")
    plt.savefig("amt_data_utilities.png")
    plt.close()

    ###
    # Get stats: number of candidates, context lengths, etc...
    # Do it ONCE for all model: these are stats of the test set itself
    ###
    print "\nCounting number of candidate responses & context lengths..."

    (candidate_counts, total_counts), (context_lengths, long_contexts, total_lengths), total, missing_at_7 = get_proportions(results)
    print "missing models when only 7 candidates are available:"
    total_miss = np.sum(missing_at_7.values())
    for model_name, miss in missing_at_7.items():
        print " - %s\tmiss: %d / %d = %f" % (model_name, miss, total_miss, miss / total_miss)
    print ""
    # plot missing models
    model_names = missing_at_7.keys()
    misses = np.array(missing_at_7.values()) / total_miss
    plt.bar(range(len(misses)), misses, tick_label=model_names)
    plt.xticks(fontsize=10, rotation=-30)
    plt.title("Proportion of models not available when we only have 7 candidates")
    plt.xlabel("Model")
    plt.ylabel("Proportion")
    plt.savefig("amt_data_miss@7.png")
    plt.close()


    print "Number of candidate responses:"
    for length, count in enumerate(candidate_counts):
        print "- #of turns with %d candidates available: %d / %d = %g" % (
            length + 1, count, total, count / total
        )
    # - measure #of_candidate avg
    print "Average number of candidates: %g" % (total_counts / total)
    # - plot proportion of #of possible candidate: #of2, #of3, ..., #of9
    candidate_counts = np.array(candidate_counts) / total
    plt.bar(range(len(candidate_counts)), candidate_counts, tick_label=range(1, len(candidate_counts)+1))
    plt.title("Number of candidate responses available")
    plt.xlabel("#of candidates")
    plt.ylabel("Proportion")
    plt.savefig("./amt_data_candidates.png")
    plt.close()

    print "\nLength of context:"
    for length, count in enumerate(context_lengths):
        print "- #of examples with %d messages in context: %d / %d = %g" % (
            length + 1, count, total, count / total
        )
    print "- #of examples with more messages in context: %d / %d = %g" % (
        long_contexts, total, long_contexts / total
    )
    # - measure context length avg
    print "Average context size: %g" % (total_lengths / total)
    # - plot proportion of context length: #of2, #of3, ..., #of10
    context_lengths.append(long_contexts)
    context_lengths = np.array(context_lengths) / total
    plt.bar(
        range(len(context_lengths)),
        context_lengths,
        tick_label=range(1, len(context_lengths))+['>']
    )
    plt.title("Context lengths")
    plt.xlabel("#of messages")
    plt.ylabel("Proportion")
    plt.savefig("./amt_data_contextlength.png")
    plt.close()

    print "done."



if __name__ == '__main__':
    main()
