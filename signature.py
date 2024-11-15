import os
import numpy as np
from google.cloud import speech
from pydub import AudioSegment
from speechbrain.inference import SpeakerRecognition
import re


voice_dir = [
    r"C:\Users\aziz\OneDrive - Institut Teccart\Bureau\medical_project\medical_application\media\doctor_voices",
    r"C:\Users\aziz\OneDrive - Institut Teccart\Bureau\medical_project\medical_application\media\patient_voices"
             ]

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r"C:\protean-set-439816-u5-9842f9fa35d4.json"


SIGNATURES_PATH = 'Voice_Signatures.npy'
EMBEDDINGS_PATH = 'Voice_Embeddings.npy'

GOOGLE_LANGUAGE_CODE = "fr-FR"

# telecharger le model
speaker_model = SpeakerRecognition.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir="pretrained_models")

# normaliser les transcriptions
def normalize_text(text):
    if text:
        text = text.lower().strip()
        text = re.sub(r'[^\w\s]', '', text)  
        text = re.sub(r'\s+', ' ', text)  
    return text

# convertir l'audio en mono
def convert_to_mono(audio_file_path):
    try:
        sound = AudioSegment.from_wav(audio_file_path)
        sound = sound.set_channels(1)  
        mono_file_path = f"mono_{os.path.basename(audio_file_path)}"
        mono_path = os.path.join(voice_dir, mono_file_path)  
        sound.export(mono_path, format="wav")
        return mono_path
    except Exception as e:
        print(f"Erreur lors de la conversion en mono: {str(e)}")
        return None

# la transcription via Google
def google_transcribe(wav_file_path):
    try:
        mono_voice_path = convert_to_mono(wav_file_path)
        if not mono_voice_path:
            raise Exception("Erreur lors de la conversion en mono")

        client = speech.SpeechClient()
        with open(mono_voice_path, "rb") as audio_file:
            content = audio_file.read()

        audio = speech.RecognitionAudio(content=content)
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            language_code=GOOGLE_LANGUAGE_CODE,
        )

        response = client.recognize(config=config, audio=audio)
        transcript = ""
        for result in response.results:
            transcript += result.alternatives[0].transcript
        return transcript
    except Exception as e:
        print(f"Erreur lors de la transcription Google: {str(e)}")
        return None

#  extraire les embeddings vocaux via SpeechBrain
def extract_voice_embedding(voice_path):
    try:
        signal = speaker_model.load_audio(voice_path)
        embedding = speaker_model.encode_batch(signal).squeeze().detach().cpu().numpy()
        return embedding
    except Exception as e:
        print(f"Erreur lors de l'extraction de l'empreinte vocale : {str(e)}")
        return None


def get_email_from_filename(filename):
    name = filename.split('.')[0]
    parts = name.split('_', 2)
    if len(parts) >= 3:
        email = f'{parts[0]}@{parts[1]}.{parts[2]}'
    else:
        email = name
    return email

if os.path.exists(SIGNATURES_PATH):
    signatures = np.load(SIGNATURES_PATH, allow_pickle=True)
else:
    signatures = np.empty((0, 2), dtype=object)  # Texte Google et email

if os.path.exists(EMBEDDINGS_PATH):
    embeddings = np.load(EMBEDDINGS_PATH, allow_pickle=True)
else:
    embeddings = np.empty((0, 2), dtype=object)  # Embedding et email


voices_list = []
emails = []

# Charger les fichiers .wav et extraire les emails
for file_name in os.listdir(voice_dir):
    if file_name.lower().endswith('.wav'):  
        voice_path = os.path.join(voice_dir, file_name)
        email = get_email_from_filename(file_name)
        voices_list.append(voice_path)
        emails.append(email)

# Extraire les caractéristiques vocales et les sauvegarder
for voice_path, email in zip(voices_list, emails):
    try:
        # Transcription via Google Cloud
        google_signature = google_transcribe(voice_path)
        print(f"Google transcription pour {email}: {google_signature}")

        # Normaliser la transcription avant de l'enregistrer
        google_signature_normalized = normalize_text(google_signature)

        # Générer l'empreinte vocale avec SpeechBrain
        embedding = extract_voice_embedding(voice_path)
        print(f"Empreinte vocale pour {email}: {embedding}")

        # Ajouter les nouvelles signatures et embeddings
        if google_signature_normalized and embedding is not None:
            signatures = np.vstack([signatures, np.array([google_signature_normalized, email], dtype=object)])
            embeddings = np.vstack([embeddings, np.array([embedding, email], dtype=object)])
        else:
            print(f"Erreur lors de l'extraction pour {email}")

    except Exception as e:
        print(f"Erreur lors du traitement de {voice_path} : {str(e)}")


np.save(SIGNATURES_PATH, signatures)
np.save(EMBEDDINGS_PATH, embeddings)
print(f"Signatures vocales enregistrées dans {SIGNATURES_PATH}")
print(f"Empreintes vocales enregistrées dans {EMBEDDINGS_PATH}")
