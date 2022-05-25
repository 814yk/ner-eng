import streamlit as st
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import TokenClassificationPipeline 
from annotated_text import annotated_text

st.header("NER Demo")


text = st.text_area('Enter text:', value="The Quad summit is taking place exactly three months after Russia invaded Ukraine — and experts say the war will likely be a major issue on the table. But not all Quad members are aligned on the conflict. The United States, Australia and Japan have all taken strong stances against the war, imposing sanctions on Russia and its oligarchs. But India, like China, has refused to condemn the invasion outright, and abstained from voting on UN resolutions demanding Moscow stop its attack. India’s response has illuminated Russia's outsized influence in Asia, where arms sales and no-strings-attached trade have allowed Moscow to exploit regional fault lines and weaker ties to the West. Why India is tied to Russia: India has long enjoyed friendly relations and a close defense relationship with Moscow; most estimates suggest more than 50% of India's military equipment comes from Russia. These supplies are vital, given India's border tensions with both China and Pakistan. Experts say India isn't looking at the situation in Ukraine in terms of their relationship with that country — it's thinking about the dangers in its own backyard.")
#reference : https://edition.cnn.com/asia/live-news/biden-asia-trip-quad-summit-05-24-22/index.html

model_name = "Jean-Baptiste/roberta-large-ner-english"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name)

nlp = TokenClassificationPipeline(task="ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple" )


colour_map = {
  'ORG': '#8fbc8f',
  'PER': '#b0c4de',
  'LOC': '#17bf33',
  'MISC': '#fffacd',
}



if text:
  ner_results = nlp(text)
  s = 0
  parsed_text = []
  for n in ner_results:
    parsed_text.append(text[s:n["start"]])
    parsed_text.append((n["word"], n["entity_group"], colour_map[n["entity_group"]]))
    s = n["end"]
  parsed_text.append(text[s:])
  annotated_text(*parsed_text)
  st.json(ner_results)


