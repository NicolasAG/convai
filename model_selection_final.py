import config
import random
import spacy
import re
import numpy as np
import json
import time
import inspect
from ranker import features
from ranker.q_networks import to_var, to_tensor, QNetwork
import torch
import torch.nn.functional as F
from Queue import Queue
import multiprocessing
import uuid
from models.wrapper import HRED_Wrapper, NQG_Wrapper, \
    DRQA_Wrapper, Topic_Wrapper, FactGenerator_Wrapper, \
    CandidateQuestions_Wrapper, DumbQuestions_Wrapper, AliceBot_Wrapper
# Dual_Encoder_Wrapper, Human_Imitator_Wrapper, HREDQA_Wrapper
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(name)s.%(funcName)s +%(lineno)s: %(levelname)-8s [%(process)d] %(message)s',
)

conf = config.get_config()


# Script to select model conversation
# Initialize all the models here and then reply with the best answer

nlp = spacy.load('en')


class ModelID:
    # Generative models
    HRED_TWITTER = 'hred-twitter'
    HRED_REDDIT = 'hred-reddit'
    NQG = 'nqg'
    # Retrieval models
    DRQA = 'drqa'
    TOPIC = 'topic_model'
    FACT_GEN = 'fact_gen'
    # Rule-based models
    CAND_QA = 'candidate_question'
    DUMB_QA = 'dumb_qa'
    ALICEBOT = 'alicebot'

    # stub to represent all allowable models
    ALL = 'all'

    # Old & bad models
    # DUAL_ENCODER = 'de'
    # HUMAN_IMITATOR = 'de-human'

    def __init__(self):
        pass


# wait time is the amount of time we wait to let the models respond.
# even if some models are taking a lot of time, do not wait for them!
WAIT_TIME = 7

# PINGBACK
# Check every time if the time now - time last pinged back of a model
# is within PING_TIME. If not, revive
PING_TIME = 60

# IPC pipes
# Parent to models
COMMAND_PIPE = 'ipc:///tmp/command.pipe'
# models to parent
BUS_PIPE = 'ipc:///tmp/bus.pipe'
# parent to bot caller
PARENT_PIPE = 'ipc:///tmp/parent_push.pipe'
# bot to parent caller
PARENT_PULL_PIPE = 'ipc:///tmp/parent_pull.pipe'

###
# Load scorer model parameters & feature list
###
with open(conf.ranker['model_final'], 'rb') as f:
    scorer_params = json.load(f)

# load the feature list used in final scorer
feature_list = []
for name, obj in inspect.getmembers(features):
    if inspect.isclass(obj) and name not in ['SentimentIntensityAnalyzer', 'Feature']:
        feature_list.append(name)

logging.info("creating scorer feature instances...")
start_creation_time = time.time()
feature_objects, feature_dim = features.initialize_features(feature_list)
assert feature_dim == 1579  # input dimension of custom encoding vector
logging.info("created all feature instances in %s sec" % (time.time() - start_creation_time))


class ModelClient(multiprocessing.Process):
    """
    Client Process for individual models. Initialize the model
    and subscribe to channel to listen for updates
    """

    def __init__(self, task_queue, result_queue, model_name):
        multiprocessing.Process.__init__(self)
        self.model_name = model_name
        self.task_queue = task_queue
        self.result_queue = result_queue

    def build(self):
        # select and initialize models
        if self.model_name == ModelID.HRED_REDDIT:
            logging.info("Initializing HRED Reddit")
            self.model = HRED_Wrapper(conf.hred['reddit_model_prefix'],
                                      conf.hred['reddit_dict_file'],
                                      ModelID.HRED_REDDIT)
        elif self.model_name == ModelID.HRED_TWITTER:
            logging.info("Initializing HRED Twitter")
            self.model = HRED_Wrapper(conf.hred['twitter_model_prefix'],
                                      conf.hred['twitter_dict_file'],
                                      ModelID.HRED_TWITTER)
        elif self.model_name == ModelID.NQG:
            logging.info("Initializing NQG")
            self.model = NQG_Wrapper('', '', ModelID.NQG)

        # elif self.model_name == ModelID.DUAL_ENCODER:
        #     logging.info("Initializing Dual Encoder")
        #     self.model = Dual_Encoder_Wrapper(conf.de['reddit_model_prefix'],
        #                                       conf.de['reddit_data_file'],
        #                                       conf.de['reddit_dict_file'],
        #                                       ModelID.DUAL_ENCODER)
        # elif self.model_name == ModelID.HUMAN_IMITATOR:
        #     logging.info("Initializing Dual Encoder on Human data")
        #     self.model = Human_Imitator_Wrapper(conf.de['convai-h2h_model_prefix'],
        #                                         conf.de['convai-h2h_data_file'],
        #                                         conf.de['convai-h2h_dict_file'],
        #                                         ModelID.HUMAN_IMITATOR)

        elif self.model_name == ModelID.DRQA:
            logging.info("Initializing DRQA")
            self.model = DRQA_Wrapper('', '', ModelID.DRQA)
        elif self.model_name == ModelID.TOPIC:
            logging.info("Initializing topic model")
            self.model = Topic_Wrapper('', '', '',
                                       conf.topic['dir_name'],
                                       conf.topic['model_name'],
                                       conf.topic['top_k'])
        elif self.model_name == ModelID.FACT_GEN:
            logging.info("Initializing fact generator")
            self.model = FactGenerator_Wrapper('', '', '')

        # elif model_name == ModelID.ECHO:
        #     logging.info("Initializing Echo")
        #     self.model = Echo_Wrapper('', '', ModelID.ECHO)
        #     self.estimate = False

        elif self.model_name == ModelID.CAND_QA:
            logging.info("Initializing Candidate Questions")
            self.model = CandidateQuestions_Wrapper('', conf.candidate['dict_file'],
                                                    ModelID.CAND_QA)
        elif self.model_name == ModelID.DUMB_QA:
            logging.info("Initializing DUMB QA")
            self.model = DumbQuestions_Wrapper('', conf.dumb['dict_file'],
                                               ModelID.DUMB_QA)
        elif self.model_name == ModelID.ALICEBOT:
            logging.info("Initializing Alicebot")
            self.model = AliceBot_Wrapper('', '', '')
        else:
            logging.error("Unrecognized model name")
            self.model = None

        self.is_running = True

        ###
        # Build NN ranker model and set its parameters
        ###
        logging.info("Building NN Scorer")
        # model_path
        scorer_path = conf.ranker['model_final'].replace('params.json', 'dqn.pt')
        # build network
        self.dqn = QNetwork(feature_dim, scorer_params['mlp_activation'], scorer_params['mlp_dropout'], 2)
        # restore old parameters
        self.dqn.load_state_dict(torch.load(scorer_path))
        # put in inference mode (Dropout OFF)
        self.dqn.eval()
        # moving networks to GPU if available
        if torch.cuda.is_available():
            logging.info("")
            logging.info("cuda available! Moving variables to cuda %d..." % args.gpu)
            self.dqn.cuda()
        logging.info("Done building NN ranker")

        self.warmup()

    def warmup(self):
        """ Warm start the models before execution """
        if self.model_name == ModelID.DRQA:
            _, _ = self.model.get_response(
                1, 'Where is Daniel?', [], nlp(unicode('Daniel went to the kitchen'))
            )
        else:
            _, _ = self.model.get_response(1, 'test_statement', [])
        self.result_queue.put(None)

    def run(self):
        """ Main running point of the process """
        # building and warming up
        self.build()
        logging.info("Built model {}".format(self.model_name))

        logging.info("Model {} listening for requests".format(self.model_name))
        while True:
            msg = self.task_queue.get()
            msg['user_id'] = msg['chat_id']
            response = ''
            context = []

            if 'control' in msg and msg['control'] == 'preprocess':
                self.model.preprocess(**msg)
            else:
                response, context = self.model.get_response(**msg)

            if len(response) > 0:
                if len(context) > 0:
                    try:
                        # Calculate features for this conversation state
                        logging.info(
                            "Start feature calculation for model {}".format(self.model_name)
                        )
                        raw_features = features.get(
                            feature_objects,
                            feature_dim,
                            msg['article_text'],
                            msg['all_context'] + [context[-1]],
                            response
                        )
                        logging.info(
                            "Done feature calculation for model {}".format(self.model_name)
                        )

                        # Run scorer and save the score in packet
                        logging.info(
                            "Start scoring the candidate response for model {}".format(self.model_name)
                        )
                        # reshape raw_features to fit the ranker format
                        assert len(raw_features) == feature_dim
                        candidate_vector = raw_features.reshape(1, feature_dim)  # ~(1, input)
                        candidate_vector = torch.Tensor(candidate_vector)
                        candidate_vector = to_var(candidate_vector)  # put on GPU if available
                        # Forward pass: predict reward
                        prediction = self.dqn(candidate_vector)  # ~(1, 2)
                        prediction = prediction[0]  # ~(2,)
                        prediction = F.softmax(prediction)
                        score = prediction.data[1]
                        logging.info(
                            "Done scoring the candidate response for model {}".format(self.model_name)
                        )
                    except:
                        logging.error("Error in estimation in model {}".format(self.model_name))
                        score = 0

                # if context is empty, set score to 0
                else:
                    score = 0

                # return the constructed message
                resp_msg = {'text': response,
                            'context': context,
                            'model_name': self.model_name,
                            'chat_id': msg['chat_id'],
                            'chat_unique_id': msg['chat_unique_id'],
                            'score': str(score)}
                self.result_queue.put(resp_msg)

            # if blank response, do not push it in the channel
            else:
                self.result_queue.put({})
        pass


class ResponseModelsQuerier(object):

    def __init__(self, model_ids):
        self.modelIDs = model_ids
        self.models = []
        for model_name in self.modelIDs:
            tasks = multiprocessing.JoinableQueue(1)
            results = multiprocessing.Queue(1)

            model_runner = ModelClient(tasks, results, model_name)
            model_runner.daemon = True
            model_runner.start()

            self.models += [{
                "model_runner": model_runner,
                "tasks": tasks,
                "results": results,
                "model_name": model_name
            }]

        # Make sure that all models are started
        for model in self.models:
            model_name = model['model_name']
            try:
                # Waiting 5 minutes for each model to initialize.
                response = model["results"].get(timeout=300)
            except Exception:
                raise RuntimeError("{} took too long to build.".format(model_name))

            if isinstance(response, Exception):
                print(
                    "\n{} Failed to initialize with error ({})."
                    "See logs in ./logs/models/".format(model_name, response)
                )
                exit(1)

    def get_response(self, msg):
        # put msg in all models
        for model in self.models:
            if 'query_models' in msg:
                if model['model_name'] in msg['query_models']:
                    model["tasks"].put(msg)
            else:
                model["tasks"].put(msg)

        # get back candidate responses from all models
        candidate_responses = {}
        for model in self.models:
            if 'query_models' in msg:
                if model['model_name'] in msg['query_models']:
                    responses = model["results"].get()
                else:
                    responses = {}
            else:
                responses = model["results"].get()

            if isinstance(responses, Exception):
                logging.error(
                    "\n{0} failed to compute response with error ({1})."
                    "\n{0} has been removed from running models.".format(model['model_name'],
                                                                         responses)
                )
                self.models.remove(model)
                if len(self.models) == 0:
                    print("All models failed. Exiting ...")
                    exit(1)
            else:
                if len(responses) > 0:
                    candidate_responses[model['model_name']] = responses
        return candidate_responses


class ModelSelectionAgent(object):
    def __init__(self):
        self.article_text = {}     # map from chat_id to article text
        self.chat_history = {}     # map from chat_id to context

        self.modelIds = [
            ModelID.HRED_TWITTER,  # general generative model on twitter data
            ModelID.HRED_REDDIT,   # general generative model on reddit data
            ModelID.NQG,  # generate a question for each sentence in the article

            ModelID.DRQA,          # return answer about the article
            ModelID.TOPIC,         # return article topic
            ModelID.FACT_GEN,      # return a fact based on conversation history

            ModelID.CAND_QA,       # return a question about an entity in the article
            ModelID.DUMB_QA,       # return predefined answer to 'simple' questions
            ModelID.ALICEBOT,      # give all responsabilities to A.L.I.C.E. ...

            # ModelID.DUAL_ENCODER,  # return a reddit turn
            # ModelID.HUMAN_IMITATOR  # return a human turn from convai round1
            # ModelID.ECHO,          # return user input
            # ModelID.FOLLOWUP_QA,   # general generative model on questions (ie: what? why? how?)
        ]

        self.response_models = ResponseModelsQuerier(self.modelIds)

    def preprocess(self, chat_id, chat_unique_id):
        _ = self.response_models.get_response({
            'control': 'preprocess',
            'article_text': self.article_text[chat_id],
            'chat_id': chat_id,
            'chat_unique_id': chat_unique_id
        })
        logging.info("preprocessed")

    def get_response(self, chat_id, text, context, control=None):
        # create a chat_id + unique ID candidate responses field
        # chat_unique_id is needed to uniquely determine the return  for each call
        chat_unique_id = str(chat_id) + '_' + str(uuid.uuid4())
        is_start = False

        logging.info(chat_id)

        # if text contains /start, don't add it to the context
        if '/start' in text:
            is_start = True
            # remove start token
            text = re.sub(r'\/start', '', text)
            # remove urls
            text = re.sub(r'https?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE)
            # save the article for later use
            self.article_text[chat_id] = text

            # initialize chat history
            self.chat_history[chat_id] = []

            # preprocessing
            # Run in separate thread so that it doesn't hold up further processing
            logging.info("Preprocessing call")
            self.preprocess(chat_id, chat_unique_id)

            logging.info("fire call")
            candidate_responses = self.response_models.get_response({
                'query_models': [ModelID.NQG, ModelID.CAND_QA],
                'article_text': self.article_text[chat_id],
                'chat_id': chat_id,
                'chat_unique_id': chat_unique_id,
                'text': '',
                'context': context,
                'all_context': self.chat_history[chat_id]
            })
            logging.info("received response")

        else:
            # making sure dict initialized
            if chat_id not in self.chat_history:
                self.chat_history[chat_id] = []

            article_text = ''
            all_context = []
            if chat_id in self.article_text:
                article_text = self.article_text[chat_id]
            if chat_id in self.chat_history:
                all_context = self.chat_history[chat_id]

            candidate_responses = self.response_models.get_response({
                'article_text': article_text,
                'chat_id': chat_id,
                'chat_unique_id': chat_unique_id,
                'text': text,
                'context': context,
                'all_context': all_context
            })

        # got the responses, now choose which one to send.
        response = None

        if is_start:
            # first interaction: select randomly either CAND_QA or NQG
            available_models = list(
                {ModelID.CAND_QA, ModelID.NQG}.intersection(
                    set(candidate_responses.keys())
                )
            )
            if len(available_models) > 0:
                selection = random.choice(available_models)
                response = candidate_responses[selection]
            # else:
            #     response = "I don't have a lot of questions about this article. Maybe you can ask me one?"

        else:
            # argmax selection: take the msg with max score

            logging.info("Removing duplicates")
            candidate_responses = self.no_duplicate(chat_id, candidate_responses)

            models = [
                ModelID.HRED_REDDIT, ModelID.HRED_TWITTER, ModelID.NQG,
                ModelID.DRQA, ModelID.TOPIC, ModelID.FACT_GEN,
                ModelID.CAND_QA, ModelID.DUMB_QA, ModelID.ALICEBOT
            ]
            available_models = list(
                set(candidate_responses.keys()).intersection(models)
            )
            if len(available_models) > 0:
                # argmax selection: take the msg with max score
                max_conf = 0
                chosen_model = available_models[0]
                for model in available_models:
                    if float(candidate_responses[model]['score']) > max_conf:
                        max_conf = float(candidate_responses[model]['score'])
                        chosen_model = model
                response = candidate_responses[chosen_model]

        # if still no response, then just send a random fact
        if not response or 'text' not in response:
            logging.warn("Failure to obtain a response, using fact gen")
            response = candidate_responses[ModelID.FACT_GEN]

        # Now we have a response, so send it back to bot host
        # add user and response pair in chat_history
        self.chat_history[response['chat_id']].append(response['context'][-1])
        self.chat_history[response['chat_id']].append(response['text'])

        response['control'] = control
        logging.info("Done selecting best model")

        return response

    def no_duplicate(self, chat_id, candidate_responses, k=5):
        del_models = []
        for model, response in candidate_responses.iteritems():
            if chat_id in self.chat_history and response['text'] in set(self.chat_history[chat_id][-k:]):
                del_models.append(model)
        for dm in del_models:
            del candidate_responses[dm]
        return candidate_responses


if __name__ == '__main__':
    model_selection_agent = ModelSelectionAgent()

    logging.info("====================================")
    logging.info("======RLLCHatBot Active=============")
    logging.info("All modules of the bot have been loaded.")
    logging.info("Thanks for your patience")
    logging.info("-------------------------------------")
    logging.info("Made with <3 in Montreal")
    logging.info("Reasoning & Learning Lab, McGill University")
    logging.info("Fall 2017")
    logging.info("=====================================")

    while True:
        user_utt = raw_input("USER : ")
        response = model_selection_agent.get_response(111, user_utt, [])
        print response
