import speech_recognition as sr
import pickle
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
import numpy as np

# Function for speech-to-text
def speech_to_text():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Please speak something...")
        recognizer.adjust_for_ambient_noise(source)
        audio_data = recognizer.listen(source)
        print("Recognizing...")
        try:
            text = recognizer.recognize_google(audio_data)
            print("You said: " + text)
            return text
        except sr.UnknownValueError:
            print("Sorry, I could not understand the audio.")
            return None
        except sr.RequestError:
            print("Sorry, my speech service is down.")
            return None

# Function to load the tokenizer
def load_tokenizer():
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    return tokenizer

# Function to load the trained model
def load_trained_model():
    model = load_model('model.keras')
    return model

# Function to predict text
def predict_text(seed_text, tokenizer, model):
    token_list = tokenizer.texts_to_sequences([seed_text])[0]
    token_list = pad_sequences([token_list], maxlen=5, padding='pre')
    predict_x = model.predict(token_list, batch_size=500, verbose=0)
    predict_x = np.argpartition(predict_x, -3, axis=1)[0][-3:]
    predictions = list(predict_x)
    predictions.reverse()
    output_words = []
    for prediction in predictions:
        for word, index in tokenizer.word_index.items():
            if prediction == index:
                output_words.append(word)
                break
    return output_words

# Main function to run the combined process
def main():
    # Step 1: Convert speech to text
    speech_text = speech_to_text()
    if speech_text:
        # Step 2: Predict text based on the converted speech
        tokenizer = load_tokenizer()
        model = load_trained_model()
        predictions = predict_text(speech_text, tokenizer, model)
        print(f"Predictions for '{speech_text}': {predictions}")

if __name__ == "__main__":
    main()
