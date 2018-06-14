import argparse
from os import listdir
from os.path import isfile, join
import re
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


WORST_R = 0.0  # worst validation F1 or TP+TN is 0.0
WORST_Q = 1.0  # worst validation loss is 1.0


def add_score(score, desc, params, optimizers, learningrates, activations, dropouts):
    """
    add score to each dictionary of params
    :param score: float value of score to add
    :param desc: string description of experiment type
    :param params: dictionary of current model params
    :param optimizers: map from optimizer to average experiment score
    :param learningrates: map from learning rate to average experiment score
    :param activations: map from activation to average experiment score
    :param dropouts: map from dropout rate to average experiment score
    """
    op = params['optimizer']
    lr = str(params['learning_rate'])
    ac = params['mlp_activation']
    do = str(params['mlp_dropout'])

    # Add to optimizer mapping
    try:
        optimizers[op][desc][0] += score  # sum all scores
        optimizers[op][desc][1] += 1.  # counts to get average
    except KeyError:
        optimizers[op][desc] = [score, 1.]
    # Add to learning rate mapping
    try:
        learningrates[lr][desc][0] += score  # sum all scores
        learningrates[lr][desc][1] += 1.  # counts to get average
    except KeyError:
        learningrates[lr][desc] = [score, 1.]
    # Add to activation mapping
    try:
        activations[ac][desc][0] += score  # sum all scores
        activations[ac][desc][1] += 1.  # counts to get average
    except KeyError:
        activations[ac][desc] = [score, 1.]
    # Add to dropout mapping
    try:
        dropouts[do][desc][0] += score  # sum all scores
        dropouts[do][desc][1] += 1.  # counts to get average
    except KeyError:
        dropouts[do][desc] = [score, 1.]


def main():
    ###
    # define regular expressions of each type of experiments
    ###
    if 'DeepQ' in args.prefix:
        regular = re.compile(r'\[exp[0-9][1-9]\]_15[0-9]{8}')  # regular experiments
        oversampled = re.compile(r'\[exp[0-9][1-9]\]os_15[0-9]{8}')  # oversampled train set
        no_custom_encs = re.compile(r'\[exp[0-9][1-9]\]noe_15[0-9]{8}')  # no custom encodings in DQN

        oversampled_no_custom_encs = re.compile(r'\[exp[0-9][1-9]\]os\+noe_15[0-9]{8}')  # oversampled train set and no custom encodings in DQN
        oversampled_no_custom_encs_and_f1 = None  # no stop of F1 for Q prediction
        oversampled_and_f1 = None  # no stop of F1 for Q prediction

    elif 'DeepR' in args.prefix:
        regular = None
        oversampled = re.compile(r'\[exp[0-9][1-9]\]os_15[0-9]{8}')  # oversampled train set
        no_custom_encs = None

        oversampled_no_custom_encs = None  # included in 'oversampled_no_custom_encs_and_F1'
        oversampled_no_custom_encs_and_f1 = re.compile(r'\[exp[0-9][1-9]\]os\+noe\+F1_15[0-9]{8}')  # oversampled train set and no custom encodings in DQN and stop on F1 score
        oversampled_and_f1 = None  # included in 'oversampled_no_custom_encs_and_F1'

    elif 'SmallQ' in args.prefix:
        regular = re.compile(r'\[exp[0-9][1-9]\]_15[0-9]{8}')  # regular experiments
        oversampled = re.compile(r'\[exp[0-9][1-9]\]os_15[0-9]{8}')  # oversampled train set
        no_custom_encs = None  # always need custom encodings for small model

        oversampled_no_custom_encs = None  # always need custom encodings for small model
        oversampled_no_custom_encs_and_f1 = None  # always need custom encodings for small model
        oversampled_and_f1 = None  # no stop of F1 for Q prediction

    elif 'SmallR' in args.prefix:
        regular = None
        oversampled = re.compile(r'\[exp[0-9][1-9]\]os_15[0-9]{8}')  # oversampled train set
        no_custom_encs = None

        oversampled_no_custom_encs = None  # always need custom encodings for small model
        oversampled_no_custom_encs_and_f1 = None  # always need custom encodings for small model
        oversampled_and_f1 = re.compile(r'\[exp[0-9][1-9]\]os\+F1_15[0-9]{8}')  # oversampled train set and stop on F1 score
    else:
        print "ERROR: Unknown prefix: %s" % args.prefix
        regular = None
        oversampled = None
        no_custom_encs = None

        oversampled_no_custom_encs = None
        oversampled_no_custom_encs_and_f1 = None
        oversampled_and_f1 = None

    ###
    # Prepare mappings for plotting average score based on parameter values
    ###
    optimizers = {'adam': {}, 'sgd': {}, 'rmsprop': {}, 'adadelta': {}}
    learningrates = {'0.01': {}, '0.001': {}, '0.0001': {}}
    activations = {'sigmoid': {}, 'prelu': {}}
    dropouts = {'0.2': {}, '0.4': {}, '0.6': {}, '0.8': {}}

    ###
    # Get best model & get scores based on parameters
    ###
    for regex, desc in zip(
            [regular, oversampled, no_custom_encs, oversampled_no_custom_encs,
             oversampled_no_custom_encs_and_f1, oversampled_and_f1],
            ['reg', 'os', 'noe', 'os+noe', 'os+noe+F1', 'os+F1']):
        if regex is not None:
            print "\nLooking for the best << %s >> model in %s..." % (desc, args.prefix)
            best_model = None
            best_score = WORST_R if 'R' in args.prefix else WORST_Q
            # loop through each file
            for fname in listdir(args.prefix):
                # catch only files that correspond to this regular expression AND finish by 'timings.json'
                if isfile(join(args.prefix, fname)) and regex.search(fname) and fname.endswith('timings.json'):
                    with open(join(args.prefix, fname), 'rb') as f:
                        timings = json.load(f)
                    with open(join(args.prefix, fname.replace('timings.json', 'params.json')), 'rb') as f:
                        params = json.load(f)

                    # Valid stats
                    valid_losses = timings['valid_losses']
                    if 'R' in args.prefix and 'F1' in timings['valid_accurs'][0]:
                        valid_f1 = [e['F1'] for e in timings['valid_accurs']]
                        #valid_f1 = [
                        #    (e['TPR'] + e['TNR']) / 2.0 for e in timings['valid_accurs']
                        #]
                    elif 'R' in args.prefix:
                        # no F1 measure so consider average between TPR and TNR
                        valid_f1 = [
                            (e['TP'] + e['TN']) / 2.0 for e in timings['valid_accurs']
                        ]
                    else:
                        valid_f1 = None

                    if valid_f1:
                        # add best valid f1 score to dictionaries of params
                        best_valid_f1 = max(valid_f1)
                        print best_valid_f1,

                        # bound worst values to WORST_R
                        if best_valid_f1 < WORST_R:
                            best_valid_f1 = WORST_R

                        add_score(best_valid_f1, desc, params, optimizers, learningrates, activations, dropouts)
                        # update best score
                        if best_valid_f1 > best_score:
                            best_score = best_valid_f1
                            best_model = fname
                    else:
                        # add best valid loss to dictionaries of params
                        best_valid_loss = min(valid_losses)
                        print best_valid_loss,

                        # bound worst values to WORST_Q
                        if best_valid_loss > WORST_Q:
                            best_valid_loss = WORST_Q

                        add_score(best_valid_loss, desc, params, optimizers, learningrates, activations, dropouts)
                        # update best score
                        if best_valid_loss < best_score:
                            best_score = best_valid_loss
                            best_model = fname

            print ""
            print "Best model: %s" % best_model
            print "with min loss / max F1 of: %g" % best_score
            with open(join(args.prefix, best_model.replace('timings.json', 'params.json')), 'rb') as f:
                params = json.load(f)
            print "with parameters: %s" % json.dumps(params, indent=2, sort_keys=True)
            print ""

    ###
    # Plotting argument mappings
    ###
    print "Plotting average valid score based on parameters..."
    # Divide by total
    for dico in [optimizers, learningrates, activations, dropouts]:
        for key, val in dico.iteritems():
            for desc in val:
                val[desc][0] /= val[desc][1]
                # val[desc] = val[desc][0]  # remove total count
        print json.dumps(dico, indent=2, sort_keys=True)
        print ""

    # set width of bar
    barwidth = 0.20
    multiplier = [0., 0.5, 1, 1.5, 2.]  # used to position xticks

    # Plot optimizers
    pos = range(len(optimizers))
    for desc, c in zip(optimizers.values()[0].keys(), ['r', 'g', 'b', 'y']):
        plt.bar(
            pos,
            [optimizers[k][desc][0] for k in optimizers.keys()],
            width=barwidth,
            label=desc,
            color=c
        )
        pos = [x + barwidth for x in pos]
    plt.legend()
    plt.title("Average valid %s based on optimizer" % ('loss' if 'Q' in args.prefix else 'F1'))
    plt.xlabel("optimizer")
    plt.xticks(
        [r + barwidth * multiplier[len(optimizers.values()[0]) - 1] for r in range(len(optimizers))],
        optimizers.keys()
    )
    plt.ylabel("score")
    plt.savefig("%s_optimizers.png" % args.prefix.replace('/', ''))
    plt.close()

    # Plot learning rates
    pos = range(len(learningrates))
    for desc, c in zip(learningrates.values()[0].keys(), ['r', 'g', 'b', 'y']):
        plt.bar(
            pos,
            [learningrates[k][desc][0] for k in learningrates.keys()],
            width=barwidth,
            label=desc,
            color=c
        )
        pos = [x + barwidth for x in pos]
    plt.legend()
    plt.title("Average valid %s based on learning rate" % ('loss' if 'Q' in args.prefix else 'F1'))
    plt.xlabel("learning rate")
    plt.xticks(
        [r + barwidth * multiplier[len(learningrates.values()[0]) - 1] for r in range(len(learningrates))],
        learningrates.keys()
    )
    plt.ylabel("score")
    plt.savefig("%s_learningrate.png" % args.prefix.replace('/', ''))
    plt.close()
    # Plot activations
    pos = range(len(activations))
    for desc, c in zip(activations.values()[0].keys(), ['r', 'g', 'b', 'y']):
        plt.bar(
            pos,
            [activations[k][desc][0] for k in activations.keys()],
            width=barwidth,
            label=desc,
            color=c
        )
        pos = [x + barwidth for x in pos]
    plt.legend()
    plt.title("Average valid %s based on activations" % ('loss' if 'Q' in args.prefix else 'F1'))
    plt.xlabel("activations")
    plt.xticks(
        [r + barwidth * multiplier[len(activations.values()[0]) - 1] for r in range(len(activations))],
        activations.keys()
    )
    plt.ylabel("score")
    plt.savefig("%s_activations.png" % args.prefix.replace('/', ''))
    plt.close()
    # Plot dropouts
    pos = range(len(dropouts))
    for desc, c in zip(dropouts.values()[0].keys(), ['r', 'g', 'b', 'y']):
        plt.bar(
            pos,
            [dropouts[k][desc][0] for k in dropouts.keys()],
            width=barwidth,
            label=desc,
            color=c
        )
        pos = [x + barwidth for x in pos]
    plt.legend()
    plt.title("Average valid %s based on dropouts" % ('loss' if 'Q' in args.prefix else 'F1'))
    plt.xlabel("dropouts")
    plt.xticks(
        [r + barwidth * multiplier[len(dropouts.values()[0]) - 1] for r in range(len(dropouts))],
        dropouts.keys()
    )
    plt.ylabel("score")
    plt.savefig("%s_dropouts.png" % args.prefix.replace('/', ''))
    plt.close()

    print "done."


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("prefix")
    args = parser.parse_args()
    main()