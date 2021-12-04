from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
 
import speech_recognition as sr
import pyttsx3 

#text to speech
from gtts import gTTS

#init the text to speech engine
engine = pyttsx3.init()

#properties of voice
voice = engine.getProperty('voices')
engine.setProperty('voice', 'english+f1')
#speech speed
rate = engine.getProperty('rate')
#engine.setProperty('rate', 125)

r = sr.Recognizer() 
def SpeakText(command):
      
    # Initialize the engine
    engine = pyttsx3.init()
    engine.say(command) 
    engine.runAndWait()
#loading the models
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-large")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-large")

# Let's chat for 5 lines
for step in range(5):
    # encode the new user input, add the eos_token and return a tensor in Pytorch
    new_user_input_ids = tokenizer.encode(input(">> User:") + tokenizer.eos_token, return_tensors='pt')

    # append the new user input tokens to the chat history
    bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1) if step > 0 else new_user_input_ids

    # generated a response while limiting the total chat history to 1000 tokens, 
    chat_history_ids = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)

    # pretty print last ouput tokens from bot
    
    output=("{}".format(tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)))
    print(output)
    engine.say(output)
    engine.runAndWait()