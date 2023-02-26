from fastai.vision.all import *
import gradio as gr

learn = load_learner("export.pkl")

categories = ("No Diabetic Retinopathy", "Diabetic Retinopathy")

def classify_image(img):
    pred,idx,probs = learn.predict(img)
    return dict(zip(categories, map(float,probs)))

image = gr.inputs.Image(shape=(192,192))
label = gr.outputs.Label()
examples = ['stage0.jpeg','stage0_2.jpeg','stage4.jpeg', 'stage4_2.jpeg']
title = 'Diabetic Retinopathy Predictor'
description = 'This app predicts diabetic retinopathy, a potentially blinding disease, from color fundus photograms. For reference only.'
article = "Author: <a href=\"https://huggingface.co/csanjay\">Sanjay Chandrasekar</a>. "

intf = gr.Interface(fn=classify_image, inputs=image, outputs=label, examples=examples, title=title, description=description, article=article)
intf.launch(inline=False)