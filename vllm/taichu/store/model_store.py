# -*- coding:utf-8 -*-
import os
import logging
import threading

logger = logging.getLogger(__name__)


class ModelStore(object):
    _instance_lock = threading.Lock()
    _init_flag = False

    def __init__(self, engine=None, tokenizer=None, max_model_len=None):
        if self._init_flag is False:
            logger.info('[ModelStore] init start ...')

            self.engine = engine
            self.tokenizer = tokenizer
            self.max_model_len = max_model_len
            logger.info("[ModelStore] self.engine:{}; self.tokenizer:{}; self.max_model_len:{};".format(
                self.engine, self.tokenizer, self.max_model_len))
            logger.info('[ModelStore] init end ...')
            self._init_flag = True
        return

    def __new__(cls, *args, **kwargs):
        if not hasattr(ModelStore, "_instance"):
            with ModelStore._instance_lock:
                if not hasattr(ModelStore, "_instance"):
                    ModelStore._instance = object.__new__(cls)
        return ModelStore._instance
