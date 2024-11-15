import os
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pydub import AudioSegment
import ffmpeg
from google.cloud import speech
from speechbrain.inference import SpeakerRecognition
import torch
import re
import torch.nn.functional as F
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r"C:\protean-set-439816-u5-9842f9fa35d4.json"

app = FastAPI()

SIGNATURES_PATH = 'Voice_Signatures.npy'
EMBEDDINGS_PATH = 'Voice_Embeddings.npy'
voice_dir = os.path.join(os.getcwd(), "doctor_voices")

if not os.path.exists(voice_dir):
    os.makedirs(voice_dir)

GOOGLE_LANGUAGE_CODE = "fr-FR"

speaker_model = SpeakerRecognition.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir="pretrained_models")

if os.path.exists(SIGNATURES_PATH):
    signatures = np.load(SIGNATURES_PATH, allow_pickle=True)
else:
    signatures = np.empty((0, 2), dtype=object)  

if os.path.exists(EMBEDDINGS_PATH):
    embeddings = np.load(EMBEDDINGS_PATH, allow_pickle=True)
else:
    embeddings = np.empty((0, 2), dtype=object)  

def normalize_text(text):
    if text:
        text = text.lower().strip() 
        text = re.sub(r'[^\w\s]', '', text)  
        text = re.sub(r'\s+', ' ', text)  
    return text

def convert_audio_to_wav(audio_file_path):
    try:
        print(f"Converting audio file {audio_file_path} to WAV format")
        sound = AudioSegment.from_file(audio_file_path)
        sound = sound.set_frame_rate(16000).set_sample_width(2).set_channels(1)

        # Save 
        wav_file_path = os.path.join(voice_dir, f"converted_{os.path.basename(audio_file_path)}")
        sound.export(wav_file_path, format="wav")
        return wav_file_path
    except Exception as e:
        print(f"Error during audio conversion to WAV: {str(e)}")
        return None

# Google transcription
def google_transcribe(wav_file_path):
    try:
        print(f"Transcribing audio file {wav_file_path} using Google Speech API")
        client = speech.SpeechClient()
        with open(wav_file_path, "rb") as audio_file:
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
        print(f"Transcription : {transcript}")
        return transcript
    except Exception as e:
        print(f"Error lors de Google transcription: {str(e)}")
        return None

def extract_voice_embedding(voice_path):
    try:
        signal = speaker_model.load_audio(voice_path)
        embedding = speaker_model.encode_batch(signal).squeeze().detach().cpu().numpy()
        print(f"Embedding extracted with shape: {embedding.shape}")
        return embedding
    except Exception as e:
        print(f"Error during voice embedding extraction: {str(e)}")
        return None



# Comparison 
def compare_embeddings(embedding1, embedding2, email):
    try:
        print(f"Comparing embeddings for email {email}")
        embedding1_tensor = torch.tensor(embedding1).unsqueeze(0)  
        embedding2_tensor = torch.tensor(embedding2).unsqueeze(0)  # dimension

        cos_sim = F.cosine_similarity(embedding1_tensor, embedding2_tensor)
        print(f"Cosine similarity score for {email}: {cos_sim.item()}")
        
        return cos_sim.item()
    except Exception as e:
        print(f"Error during embedding comparison: {str(e)}")
        return None


@app.post("/verify-voice-id/")
async def verify_voice_id(email: str = Form(...), file: UploadFile = File(...)):
    print(f"Received verification request for email: {email} with file: {file.filename}")
    try:
        
        audio_path = os.path.join(voice_dir, f"temp_{file.filename}")
        print(f"Saving audio file to {audio_path}")
        with open(audio_path, "wb") as buffer:
            buffer.write(file.file.read())

        # Convert to wav
        wav_audio_path = convert_audio_to_wav(audio_path)
        if not wav_audio_path:
            raise HTTPException(status_code=500, detail="Error converting audio file to valid WAV format.")

        # Google transcription
        transcript = google_transcribe(wav_audio_path)
        transcript_normalized = normalize_text(transcript)

        # embeddings
        embedding = extract_voice_embedding(wav_audio_path)

        if transcript_normalized and embedding is not None:
            for google_sig, stored_email in signatures:
                if stored_email == email and transcript_normalized == normalize_text(google_sig):
                    print(f"Matching transcript found for {email}")
                    for stored_embedding, embedding_email in embeddings:
                        if embedding_email == email:
                            stored_embedding = np.array(stored_embedding)
                            score = compare_embeddings(embedding, stored_embedding, email)

                            if score > 0.36: 
                                print(f"Authentication succeeded for {email} with score: {score}")
                                return {"message": "Double authentication successful."}
                            else:
                                print(f"Authentication failed for {email} with score: {score}")
        
        raise HTTPException(status_code=401, detail="Text or voice not recognized.")
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during verification: {str(e)}")


@app.post("/add-voice-signature/")
async def add_voice_signature(file: UploadFile = File(...), email: str = Form(...)):
    try:
        
        audio_path = os.path.join(voice_dir, f"temp_{file.filename}")
        with open(audio_path, "wb") as buffer:
            buffer.write(file.file.read())

        wav_audio_path = convert_audio_to_wav(audio_path)
        if not wav_audio_path:
            raise HTTPException(status_code=500, detail="Error converting audio file to valid WAV format.")

        transcript = google_transcribe(wav_audio_path)
        if not transcript:
            raise HTTPException(status_code=500, detail="Error during Google transcription")

        transcript_normalized = normalize_text(transcript)

        embedding = extract_voice_embedding(wav_audio_path)
        if embedding is None:
            raise HTTPException(status_code=500, detail="Error extracting voice embeddings")

        global signatures, embeddings
        new_signature = np.array([[transcript_normalized, email]], dtype=object)
        new_embedding = np.array([[embedding, email]], dtype=object)

        signatures = np.vstack([signatures, new_signature])
        embeddings = np.vstack([embeddings, new_embedding])

        np.save(SIGNATURES_PATH, signatures)
        np.save(EMBEDDINGS_PATH, embeddings)

        os.remove(audio_path)

        return {"message": f"Voice signature and embeddings successfully added for email: {email}"}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error adding voice signature: {str(e)}")

