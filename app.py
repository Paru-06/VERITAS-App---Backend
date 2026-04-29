import gradio as gr
from api import predict

def detect(news):
    r = predict(news)
    return f"{r['prediction']} | Confidence: {r['confidence']}"

gr.Interface(
fn=detect,
inputs="textbox",
outputs="text",
title="VERITAS Fake News Detector"
).launch()