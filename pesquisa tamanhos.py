import requests
import pandas as pd
import matplotlib.pyplot as plt
import os
import json
import numpy as np
from numpy import *
from scipy.signal import medfilt
from wikiLib import *

os.system('clear')

title = "Standard Model"
#title = "Donald Trump"
#title = "Katseye"
#title = "AJR"
#title = "Coronavirus"

print("Calculando dados para o artigo " + title)
wp = Wikipedia()

# Fetch data
fields = "timestamp, size"
article = wp.get_article(title, fields)

print("data received")


timestamps = article.timestamps
sizes = article.size
sizes_filtered = medfilt(sizes, kernel_size=5)
edits_per_day = article.resample("timestamps", 30, "count")

edit_size = article.resample("size", 2, "diff_median")

fig, graphs = plt.subplots(3, 1, figsize=(8, 10), sharex=True)

graphs[0].plot(timestamps, sizes, label="tamanho do artigo", lw = 1, ls = "dotted")
graphs[0].plot(timestamps, sizes_filtered, label="tamanho do artigo suavizado", lw = 2)
graphs[0].set_title("Tamanho por data")
graphs[0].legend()

graphs[1].plot(edits_per_day.index, edits_per_day.values, label="edições a cada 30 dias", lw = 2)
graphs[1].set_title("edições a cada 30 dias")

graphs[1].legend()

graphs[2].plot(edit_size.index, edit_size.values, label="tamanho médio de edições", lw = 2)
graphs[2].set_title("tamanho médio de edição por data")

graphs[2].legend()

fig.suptitle("Pesquisa numérica do artigo " + title)

plt.savefig("size_plot.png")