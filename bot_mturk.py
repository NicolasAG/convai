import zmq
import sys
reload(sys)
sys.setdefaultencoding('utf8')
import requests
import os
import json
import time
import random
import collections
import config
conf = config.get_config()
import random
import emoji
import numpy as np
# import storage
from model_selection_mturk import ModelID, ModelSelectionAgent
import multiprocessing
from Queue import Queue
from threading import Thread
import logging
from datetime import datetime
import traceback
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(name)s.%(funcName)s +%(lineno)s: %(levelname)-8s [%(process)d] %(message)s',
)

"""
Copyright 2017 Reasoning & Learning Lab, McGill University

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

# MTurk version of the bot

MAX_CONTEXT = 3

chat_history = {}
chat_timing = {}  # just for testing response time

# Queues
processing_msg_queue = Queue()
outgoing_msg_queue = Queue()  # multiprocessing.JoinableQueue()

# Pipes
# bot to wrapper pull pipe
REQUEST_PIPE = 'ipc:///tmp/request.pipe'
# bot to wrapper REQ/RESP pipe
RESPONSE_PIPE = 'ipc:///tmp/response.pipe'

# Utils


def mogrify(topic, msg):
    """ json encode the message and prepend the topic """
    return str(topic) + ' ' + json.dumps(msg)


def demogrify(topicmsg):
    """ Inverse of mogrify() """
    json0 = topicmsg.find('{')
    topic = topicmsg[0:json0].strip()
    msg = json.loads(topicmsg[json0:])
    return topic, msg


class ChatState:
    START = 0     # when we received `/start`
    END = 1       # when we received `/end`
    CHATTING = 2  # all other times
    CONTROL = 3   # for control msgs


class ConvAIRLLBot:

    def __init__(self):
        self.chat_id = None
        self.observation = None
        self.ai = {}

    def observe(self, m):
        chat_id = m['chat']['id']
        state = ChatState.CHATTING  # default state
        # New chat:
        if chat_id not in self.ai:
            if m['text'].startswith('/start '):
                self.ai[chat_id] = {}
                self.ai[chat_id]['chat_id'] = chat_id
                self.ai[chat_id]['observation'] = m['text']
                # changed from deque since it is not JSON serializable
                self.ai[chat_id]['context'] = []
                self.ai[chat_id]['allowed_model'] = ModelID.ALL
                logging.info("Start new chat #%s" % self.chat_id)
                state = ChatState.START  # we started a new dialogue
                chat_history[chat_id] = []
                logging.info("started new chat with {}".format(chat_id))
            else:
                logging.info("chat not started yet. Ignore message")
        # Not a new chat:
        else:
            # Finished chat
            if m['text'] == '/end':
                logging.info("End chat #%s" % chat_id)
                processing_msg_queue.put(
                    {'control': 'clean', 'chat_id': chat_id})
                del self.ai[chat_id]
                state = ChatState.END  # we finished a dialogue
            # TODO: Control statement for allowed models
            # Statement could start with \model start <model_name>
            # End with \model end <model_name>
            elif m['text'].startswith("\\model"):
                controls = m['text'].split()
                if controls[1] == 'start':
                    # always provide a valid model name for debugging.
                    self.ai[chat_id]['allowed_model'] = controls[2]
                    logging.info("Control msg accepted, selecting model {}"
                                 .format(controls[2]))
                else:
                    self.ai[chat_id]['allowed_model'] = ModelID.ALL
                    logging.info(
                        "Control msg accepted, resetting model selection")
                state = ChatState.CONTROL
            # Continue chat
            else:
                self.ai[chat_id]['observation'] = m['text']
                logging.info("Accept message as part of chat #%s" % chat_id)
        return chat_id, state

    def act(self, chat_id, state,  m):
        data = {}
        if chat_id not in self.ai:
            # Finish chat:
            if m['chat']['id'] == chat_id and m['text'] == '/end':
                logging.info("Decided to finish chat %s" % chat_id)
                data['text'] = '/end'
                data['evaluation'] = {
                    'quality': 5,
                    'breadth': 5,
                    'engagement': 5
                }
                if chat_id in chat_history:
                    # storage.store_data(chat_id, chat_history[chat_id])
                    del chat_history[chat_id]
                outgoing_msg_queue.put(
                    ({'data': data, 'chat_id': chat_id}, {}))
                return
            else:
                logging.info("Dialog not started yet. Do not act.")
                return

        if self.ai[chat_id]['observation'] is None:
            logging.info("No new messages for chat #%s. Do not act." %
                         self.chat_id)
            return

        model_name = 'none'
        policyID = -1
        if state != ChatState.CHATTING:
            if state == ChatState.CONTROL:
                text = "--- Control command received ---"
            # if state == ChatState.START:
            #    text = "Hello! I hope you're doing well. I am doing fantastic today! Let me go through the article real quick and we will start talking about it."
            # push this response to `outgoing_msg_queue`
            # outgoing_msg_queue.put(
            #    ({'text': text, 'chat_id': chat_id,
            #        'model_name': model_name, 'policyID': policyID}, {}))
        else:
            # send the message to process queue for processing
            processing_msg_queue.put({
                'chat_id': chat_id,
                'text': self.ai[chat_id]['observation'],
                'context': m['context'],
                'allowed_model': self.ai[chat_id]['allowed_model']
            })


# Initialize
BOT_ID = conf.bot_token  # !!!!!!! Put your bot id here !!!!!!!

if BOT_ID is None:
    raise Exception('You should enter your bot token/id!')

BOT_URL = os.path.join(conf.bot_endpoint, BOT_ID)

bot = ConvAIRLLBot()
model_selection_agent = None


def producer():
    """ call model selection agent here
    """
    while True:
        msg = processing_msg_queue.get()
        producer_process(msg)
        # m_p = multiprocessing.Process(target=producer_process, args=(msg,outgoing_msg_queue,
        #    model_selection_agent,))
        #m_p.daemon = True
        # m_p.start()
        processing_msg_queue.task_done()


def producer_process(msg):
    if 'text' in msg and 'chat_id' in msg and 'context' in msg:
        time_now = datetime.now()
        responses = model_selection_agent.get_response(
            msg['chat_id'], msg['text'], msg['context'])
        outgoing_msg_queue.put(responses)


def response_receiver(telegram=True):
    """Receive response from REQUEST pipe.
       Make its own thread for clarity
       Incoming message is the following dict:
       { 'prev_response' : text, 'message' : text }
    """
    context = zmq.Context()
    socket = context.socket(zmq.PULL)
    socket.bind(REQUEST_PIPE)
    while True:
        m = socket.recv_json()
        state = ChatState.START  # assume new chat all the time
        # will become false when we call bot.observe(m),
        # except when it's really a new chat
        while state == ChatState.START:
            logging.info("Process message %s" % m)
            # return chat_id & the dialogue state
            chat_id, state = bot.observe(m)
            bot.act(chat_id, state, m)


def reply_sender():
    """ Send reply to Telegram or console
        Thread: Read from `outgoing_msg_queue` and send to server
    """
    context = zmq.Context()
    socket = context.socket(zmq.PUB)
    socket.bind(RESPONSE_PIPE)
    while True:
        package = outgoing_msg_queue.get()
        msg = package
        # preserving the state
        # for key,value in cache.iteritems():
        #    setattr(model_selection_agent, key, value)
        outgoing_msg_queue.task_done()
        if 'chat_id' not in msg[0]:
            continue
        chat_id = msg[0]['chat_id']
        message = {
            'chat_id': chat_id
        }
        message['responses'] = msg
        # if chat has ended, then no need to put the chat history
        # if chat_id in chat_history:
        #     chat_history[chat_id].append(data)

        logging.info("Publish responses")
        socket.send(mogrify(chat_id, message))


def stop_app():
    """Broadcast stop signal
    """
    processing_msg_queue.put({'control': 'exit'})


def test_app():
    """ Perform heavy testing on the app.
    Send msgs and log the response avg response time.
    """
    chat_id = random.randint(1, 100000)
    text = random.choice(['ok, how was it', 'wow good observation',
                          'where did it happen?', 'can you explain?', 'ok', 'hmm', 'good',
                          'i dont like this', 'interesting', 'where did you see this?',
                          'who made it?', '\start McGill University is a public research organization'])
    context = ['preset context']
    allowed_model = 'all'
    processing_msg_queue.put({
        'chat_id': chat_id,
        'text': text,
        'context': context,
        'allowed_model': allowed_model,
        'control': 'test'
    })
    chat_timing[chat_id] = datetime.now()


if __name__ == '__main__':
    """ Start the threads.
    1. Response reciever thread
    2. Producer thread. bot -> model_selection
    3. Consumer thread. model_selection -> bot
    4. Reply thread. bot -> Telegram
    """
    MODE = 'production'  # can be 'test' or anything else
    model_selection_agent = ModelSelectionAgent()

    response_receiver_thread = Thread(target=response_receiver, args=(True,))
    response_receiver_thread.daemon = True
    response_receiver_thread.start()
    producer_thread = Thread(target=producer)
    producer_thread.daemon = True
    producer_thread.start()
    reply_thread = Thread(target=reply_sender)
    reply_thread.daemon = True
    reply_thread.start()

    logging.info("====================================")
    logging.info("======RLLCHatBot Active=============")
    logging.info("All modules of the bot have been loaded.")
    logging.info("Thanks for your patience")
    logging.info("-------------------------------------")
    logging.info("Made with <3 in Montreal")
    logging.info("Reasoning & Learning Lab, McGill University")
    logging.info("Fall 2017")
    logging.info("=====================================")

    try:
        while True:
            if MODE == 'test':
                test_app()
                if random.choice([True, False]):
                    test_app()
                test_app()
                # test_app()
            time.sleep(10)
    except (KeyboardInterrupt, SystemExit):
        logging.info("Stopping model response selector")
        stop_app()
        logging.info("Closing app")
