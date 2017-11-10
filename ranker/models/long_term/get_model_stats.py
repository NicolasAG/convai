import numpy as np
import cPickle as pkl
import argparse

def main(args):
    for model_id in args.ids:
        with open("./%sargs.pkl" % model_id, 'rb') as handle:
            args = pkl.load(handle)
        n_folds = len(args[0][0])

        with open('./%stimings.pkl' % model_id, 'rb') as handle:
            train_accuracies, valid_accuracies = pkl.load(handle)
        max_trains = [max(accs) for accs in train_accuracies]
        max_valids = [max(accs) for accs in valid_accuracies]

        print "max train accuracies: %s" % max_trains
        print "max valid accuracies: %s" % max_valids
        recent_max_trains = [max(train_accuracies[i]) for i in range(-n_folds, 0)]
        recent_max_valids = [max(valid_accuracies[i]) for i in range(-n_folds, 0)]

        print "%s \t avg. train: %g \t avg. valid: %g \t args: %s" % (model_id, np.mean(recent_max_trains), np.mean(recent_max_valids), args[1:])

        with open('./%svalid%g.txt' % (model_id, np.mean(recent_max_valids)), 'w') as handle:
            handle.write("model %s\n" % model_id)
            handle.write("avg. train: %g\n" % np.mean(recent_max_trains))
            handle.write("avg. valid: %g\n" % np.mean(recent_max_valids))
            handle.write("args: %s\n" % args[1:])
            handle.write("features:\n%s\n" % args[0][-1])
            handle.write("-------------\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("ids", nargs='+', type=str, help="List of model ids to get validation score from timings")
    args = parser.parse_args()
    main(args)

