#Arquivo contendo todas as bibliotecas utilizadas no projeto
import numpy as np
import os
import pandas as pd
import platform
import re
from joblib import dump, load
from flask import Flask, request, jsonify, render_template
from gensim.models import Word2Vec
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier, Perceptron, SGDClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from unidecode import unidecode