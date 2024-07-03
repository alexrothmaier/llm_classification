#read data from csv
import sys
sys.path.append('..')
import logging
logging.basicConfig(level=logging.WARNING)

from src.llm_alex import Llama
from langchain_core import pydantic_v1
from langchain_core.runnables.base import RunnableParallel, RunnableLambda
from langchain.output_parsers.retry import RetryOutputParser


import numpy
import pandas as pd
import random 
import tqdm 
import time
import wandb

from sklearn.metrics import classification_report

import matplotblib.pyplot as plt

llm = Llama()
print("test")