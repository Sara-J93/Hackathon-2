print("💡 Fichier app.py bien modifié !")
# ----------------------------------------
# 🌟 Assistant IA Empathique - app.py
# ----------------------------------------

# Imports essentiels pour ton app
import streamlit as st
from transformers import pipeline, AutoTokenizer as FalconTokenizer, AutoModelForCausalLM as FalconModel
import torch  # 🔥 Garde torch uniquement car il est utilisé pour Falcon



from transformers import pipeline

# On charge un modèle public déjà entraîné sur les émotions
emotion_classifier = pipeline("text-classification", model="bhadresh-savani/distilbert-base-uncased-emotion")

# ✔ Nouvelle fonction de détection :
def detect_emotion(text):
    result = emotion_classifier(text)[0]
    return result["label"].lower()

# 📌 Liste des étiquettes d’émotions
emotion_labels = ["sadness", "joy", "love", "anger", "fear", "surprise"]

# ----------------------------------------
# 🤖 Moteur de réponse empathique - Falcon 7B
# ----------------------------------------

# from transformers import AutoTokenizer as FalconTokenizer
# from transformers import AutoModelForCausalLM as FalconModel

# # Charger Falcon 7B Instruct
# falcon_model_name = "tiiuae/falcon-7b-instruct"
# falcon_tokenizer = FalconTokenizer.from_pretrained(falcon_model_name)



# import os
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"  # ✅ Prevent CUDA OOM

# from transformers import AutoTokenizer as FalconTokenizer
# from transformers import AutoModelForCausalLM as FalconModel

# falcon_model_name = "tiiuae/falcon-7b-instruct"

# falcon_tokenizer = FalconTokenizer.from_pretrained(falcon_model_name)
# falcon_model = FalconModel.from_pretrained(
#     falcon_model_name,
#     device_map="auto",  # ✅ Spread layers across GPU and CPU if needed
#     trust_remote_code=True,  # ✅ Required for Falcon
#     torch_dtype=torch.float16  # ✅ Half precision to save GPU memory
# )

from transformers import AutoTokenizer as FalconTokenizer
from transformers import AutoModelForCausalLM as FalconModel
import torch

falcon_model_name = "tiiuae/falcon-7b-instruct"
device = "cuda" if torch.cuda.is_available() else "cpu"

falcon_tokenizer = FalconTokenizer.from_pretrained(falcon_model_name)
falcon_model = FalconModel.from_pretrained(
    falcon_model_name,
    device_map="auto",            # Spread across GPU/CPU
    torch_dtype=torch.float16,    # Use FP16 to save VRAM
    trust_remote_code=True,
    low_cpu_mem_usage=True        # Optimize RAM usage
)

falcon_model.eval()

# !pip install accelerate

# Fonction pour générer une réponse empathique
def generate_empathic_response_llm(emotion_label):
    prompt = (
        f"You are a kind, caring and empathetic assistant. "
        f"A user is feeling {emotion_label}. "
        f"Write a short, warm and emotionally supportive sentence."
    )

    inputs = falcon_tokenizer(prompt, return_tensors="pt").to(device)
    outputs = falcon_model.generate(
        **inputs,
        max_length=60,
        do_sample=True,
        top_k=50,
        top_p=0.9,
        temperature=0.7,
        pad_token_id=falcon_tokenizer.eos_token_id
    )

    response = falcon_tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Nettoyer la sortie
    if prompt in response:
        response = response.replace(prompt, "").strip()
    return response.strip()

# ----------------------------------------
# 🎯 Fonction de détection d’émotion
# ----------------------------------------

# def detect_emotion(text):
#     inputs = emotion_tokenizer(text, return_tensors="pt", truncation=True, padding=True)
#     inputs = {k: v.to(emotion_model.device) for k, v in inputs.items()}
#     outputs = emotion_model(**inputs)
#     predicted_label = torch.argmax(outputs.logits, dim=1).item()
#     return emotion_labels[predicted_label]

# ----------------------------------------
# 💻 Interface utilisateur Streamlit
# ----------------------------------------

st.set_page_config(page_title="IA Empathique", page_icon="🤖")
st.title("💬 Assistant IA Empathique")
st.markdown("Entrez un message ci-dessous. L'IA détectera votre émotion et vous répondra avec bienveillance.")

# Zone de texte utilisateur
user_input = st.text_area("✍️ Votre message :", height=150)

# Bouton de validation
if st.button("Analyser & Répondre"):
    if not user_input.strip():
        st.warning("⛔ Veuillez écrire un message.")
    else:
        with st.spinner("🧠 Analyse de l'émotion..."):
            emotion = detect_emotion(user_input)
            response = generate_empathic_response_llm(emotion)

        # Affichage des résultats
        st.success(f"🧠 Émotion détectée : **{emotion.upper()}**")
        st.info(f"🤖 Réponse de l'IA : *{response}*")
