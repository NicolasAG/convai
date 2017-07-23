# Script to select model conversation
# Initialize all the models here and then reply with the best answer
# some logic-foo need to be done here.
import config
conf = config.get_config()
from models.wrapper import HRED_Wrapper, Dual_Encoder_Wrapper, HREDQA_Wrapper, CandidateQuestions_Wrapper, DumbQuestions_Wrapper, DRQA_Wrapper
import random
import copy
import spacy

nlp = spacy.load('en')


class ModelSelection(object):

    def __init__(self):
        self.hred_model_twitter = None
        self.hred_model_reddit = None
        self.article_text = {}
        self.candidate_model = {}

    def initialize_models(self):
        self.hred_model_twitter = HRED_Wrapper(conf.hred['twitter_model_prefix'], conf.hred['twitter_dict_file'], 'hred-twitter')
        self.hred_model_reddit = HRED_Wrapper(conf.hred['reddit_model_prefix'], conf.hred['reddit_dict_file'], 'hred-reddit')
        self.de_model_reddit = Dual_Encoder_Wrapper(conf.de['reddit_model_prefix'], conf.de['reddit_data_file'], conf.de['reddit_dict_file'], 'de-reddit')
        self.qa_hred = HREDQA_Wrapper(conf.followup['model_prefix'],conf.followup['dict_file'],'followup_qa')
        self.dumb_qa = DumbQuestions_Wrapper('',conf.dumb['dict_file'],'dumb_qa')
        self.drqa = DRQA_Wrapper('','','drqa')
        # warmup the models before serving
        r,_ = self.hred_model_twitter.get_response(1, 'test statement', [])
        r,_ = self.hred_model_reddit.get_response(1, 'test statement', [])
        r,_ = self.de_model_reddit.get_response(1, 'test statement', [])
        r,_ = self.qa_hred.get_response(1, 'test statement', [])
        r,_ = self.drqa.get_response(1,'Where is Daniel?',[],'Daniel went to the kitchen')

    def get_response(self,chat_id,text,context):
        # if text containes /start, dont add it to the context
        if '/start' in text:
            # save the article for later use
            self.article_text[chat_id] = nlp(unicode(text))
            self.candidate_model[chat_id] = CandidateQuestions_Wrapper('',self.article_text[chat_id],
                    conf.candidate['dict_file'],'candidate_question')
            # generate first response or not?
            # with some randomness generate the first response or leave blank
            if random.choice([True,False]):
                #resp = 'Nice article, what is it about?'
                resp,context = self.candidate_model[chat_id].get_response(chat_id,'',context)
                #context.append('<first_speaker>' + resp + '</s>')
            else:
                resp = ''
            return resp,context
        # chat selection logic
        # if text contains a question, do not respond with a question (followup)
        # if query falls under dumb questions, respond appropriately
        if self.dumb_qa.isMatch(text):
            resp,context = self.dumb_qa.get_response(chat_id,text,context)
            return resp,context

        # if text contains question, run DRQA
        if '?' in text:
            resp,context = self.drqa.get_response(chat_id,text,context,article=self.article_text[chat_id].text)
            return resp,context
        outputs = []
        origin_context = copy.deepcopy(context)
        resp1,cont1 = self.hred_model_twitter.get_response(chat_id, text, origin_context, self.article_text.get(chat_id,''))
        outputs.append((resp1,cont1))
        origin_context = copy.deepcopy(context)
        resp2,cont2 = self.hred_model_reddit.get_response(chat_id, text, origin_context, self.article_text.get(chat_id,''))
        outputs.append((resp2,cont2))
        origin_context = copy.deepcopy(context)
        resp3,cont3 = self.de_model_reddit.get_response(chat_id, text, origin_context, self.article_text[chat_id])
        outputs.append((resp3,cont3))
        origin_context = copy.deepcopy(context)
        if '?' not in text:
            resp4,cont4 = self.qa_hred.get_response(chat_id, text, origin_context, self.article_text.get(chat_id,''))
            outputs.append((resp4,cont4))

        # chat selection logic
        # for now, select in random
        ch = random.choice(range(len(outputs)))

        return outputs[ch]
