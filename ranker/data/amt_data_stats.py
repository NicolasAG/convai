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
    data = {}  # map from model_name to number of time available & number of time used
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

            # loop through each candidate option
            for option_idx, option in turn['options'].items():
                # don't consider the /end option
                if option_idx == '/end':
                    continue

                # reset some model names
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

                # store model name
                if model_name not in data:
                    data[model_name] = {
                        'available': 0.,
                        'used': 0.
                    }

                # always add one when available
                data[model_name]['available'] += 1

                # if chosen, add one to 'used' count
                if choice_idx == option_idx:
                    data[model_name]['used'] += 1

                # increment example counter
                n_ex += 1

            # end of options
        # end of conversation
    # end of accepted HITs
    return data, n_ex


def main():
    # connect to database
    print "connect to database & query approved conversations..."
    client = pymongo.MongoClient(DB_CLIENT, DB_PORT)
    db = client.rllchat_mturk

    # get all accepted conversations
    results = list(db.dialogs.find({'review': 'accepted'}))
    print "got %d conversations." % len(results)

    # get quality of all conversations
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

    # get model availabilities
    model_usage, n_ex = get_model_usage(results)
    for model_name, usage in model_usage.items():
        print " - %s\tavailability: %d / %d = %f" % (model_name[:], usage['available'], n_ex, usage['available'] / n_ex)
    print ""

    # plot availabilities
    model_names = model_usage.keys()
    availabilities = [usage['available'] / n_ex for usage in model_usage.values()]
    plt.bar(range(len(availabilities)), availabilities, tick_label=model_names)
    plt.xticks(fontsize=10, rotation=-30)
    plt.title("Model availabilities")
    plt.xlabel("Model")
    plt.ylabel("Proportion")
    plt.savefig("amt_data_availabilities.png")
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
    plt.title("Model utilities")
    plt.xlabel("Model")
    plt.ylabel("Proportion")
    plt.savefig("amt_data_utilities.png")
    plt.close()



if __name__ == '__main__':
    main()
