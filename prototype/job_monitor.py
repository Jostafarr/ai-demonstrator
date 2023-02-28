import streamlit as st
import os
import sys
import pandas as pd
import numpy as np

from utils import JobMonitor, state_change


sys.path.append("/Users/jostgotte/Documents/Uni/WS2223/rtiai/TIE/")


monitor = JobMonitor()

st.title('Competitor Job Monitor')

st.write('Diese Apps überwacht die Jobangebote von Wettbewerbern und informiert Sie, wenn neue Stellenangebote veröffentlicht werden.')



st.text_input(label="Bitte geben Sie uns die URL der Ziel-Website",  key='url')
st.text_input(label="Welche Frage sollen wir für Sie beantworten?",  key="question")

if st.button(label="Lade neuen Jobeintrag"):
    state_change()   


if st.session_state.url and st.session_state.question:
    monitor.monitor(url=st.session_state.url, question=st.session_state.question)



