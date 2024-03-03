# Packages for Whisper model
from whisper import whisper

# Packages for ChatYuan model
import torch
from transformers import AutoTokenizer

# Packages for YourTTS model
import TTS
from TTS.api import TTS

''' Default Settings '''
device = torch.device('cuda') # set as GPU for faster inference
# Settings of Whisper
input_audio_path = r"录音1.m4a"
s2t_transcribe_mode = 'detect_language' # ['directly_transcribe', 'detect_language']
# Settings of ChatYuan

# Settings of YourTTS
clone_speaker_voice = r'clone_speaker.wav'
output_audio_path = r"output.wav"

''' (1) Whisper: Transcribe Speech to Text '''
# load Whisper model
print('--(1) Loading Whisper Model for Speech2Text...')
whisper_model = whisper.load_model("small")
# whisper_model.to(device) # Whisper on CPU, not on GPU for now
print('--Successfully Loaded Whisper Model!')

print('--Reading Input Audio: ' + '\033[1;36m'+input_audio_path+'\033[0m')
if s2t_transcribe_mode == 'directly_transcribe': # directly transcribe audio to text
    result = whisper_model.transcribe(input_audio_path)
    # transcribe() reads the entire file and processes the audio with a sliding 30-second window,
    # and performing autoregressive sequence-to-sequence predictions on each window
    whisper_text_output = result["text"]
    detected_language = result["language"]
elif s2t_transcribe_mode == 'detect_language': # detect spoken language and then transcribe to text
    # load audio and pad/trim it to fit 30 seconds
    audio = whisper.load_audio(input_audio_path)
    audio = whisper.pad_or_trim(audio)
    # make log-Mel spectrogram and move to the same device as the model
    mel = whisper.log_mel_spectrogram(audio).to(whisper_model.device)

    # detect the spoken language
    _, probs = whisper_model.detect_language(mel)
    detected_language = max(probs, key=probs.get)
    print(f"--Detected Language: \033[1;36m {detected_language}\033[0m")

    # decode the audio
    options = whisper.DecodingOptions()
    # options = whisper.DecodingOptions(fp16=False)
    result = whisper.decode(whisper_model, mel, options)
    whisper_text_output = result.text

# print the recognized text
print('--Recognition Text of Input Audio:')
print('  \033[1;36m'+whisper_text_output+'\033[0m')

''' (2) ChatYuan: Text to Text Dialogue Chat '''
def preprocess(text):
  text = text.replace("\n", "\\n").replace("\t", "\\t")
  return text

def postprocess(text):
  return text.replace("\\n", "\n").replace("\\t", "\t")

def answer(text, tokenizer, model, sample=True, top_p=1, temperature=0.7):
  '''
      sample：是否抽样。生成任务，可以设置为True;
      top_p：0-1之间，生成的内容越多样
  '''
  text = preprocess(text)
  encoding = tokenizer(text=[text], truncation=True, padding=True,
                                max_length=768, return_tensors="pt").to(device)
  if not sample:
    out = model.generate(
        **encoding, return_dict_in_generate=True, output_scores=False,
        max_new_tokens=512, num_beams=1, length_penalty=0.6)
  else:
    out = model.generate(
        **encoding, return_dict_in_generate=True, output_scores=False,
        max_new_tokens=512, do_sample=True, top_p=top_p, temperature=temperature, no_repeat_ngram_size=3)
  out_text = tokenizer.batch_decode(out["sequences"], skip_special_tokens=True)
  return postprocess(out_text[0])

print('--(2) Loading ChatYuan Model for Dialogue Generation...')
# load ChatYuan model
from transformers import T5Tokenizer, T5ForConditionalGeneration
chatyuan_tokenizer = T5Tokenizer.from_pretrained("ClueAI/ChatYuan-large-v1")
chatyuan_model = T5ForConditionalGeneration.from_pretrained("ClueAI/ChatYuan-large-v1")
chatyuan_model.to(device)
print('--Successfully Loaded ChatYuan Model!')

print('--Reading Input Question: ' + '\033[1;36m'+whisper_text_output+'\033[0m')
chatyuan_input_list = []
chatyuan_input_list.append(whisper_text_output)
for i, input_text in enumerate(chatyuan_input_list):
  input_text = "用户：" + input_text + "\n小元："
  print(f"--ChatYuan Answer{i}: ")
  # print(f"ChatYuan示例{i}".center(50, "="))
  output_text = answer(input_text, chatyuan_tokenizer, chatyuan_model)
  print('\033[1;36m' +f"{input_text}{output_text}"+'\033[0m')
chatyuan_output_text = output_text

''' (3) YourTTS: Transcribe Text to Speech '''
# load YourTTS model
print('--(3) Loading YourTTS Model for Text2Speech...')
# list available 🐸TTS models and choose the first one
# tts_model_name = TTS.list_models()[0] # as YourTTS 'tts_models/multilingual/multi-dataset/your_tts'
# tts_model_name = TTS.list_models()[28] # for Chinese 'tts_models/zh-CN/baker/tacotron2-DDC-GST'
# init TTS
# tts_model = TTS(tts_model_name, gpu=True)
if detected_language=='zh':
    tts_model = TTS(model_path=r".\Model_Weights\tts_models--zh-CN--baker--tacotron2-DDC-GST\model_file.pth",
                    config_path=r".\Model_Weights\tts_models--zh-CN--baker--tacotron2-DDC-GST\config.json",
                    progress_bar=False, gpu=True)
elif detected_language=='en':
    tts_model = TTS(model_name="tts_models/multilingual/multi-dataset/your_tts", progress_bar=False, gpu=True)
print('--Successfully Loaded YourTTS Model!')

print('--Reading Input Text: ' + '\033[1;36m'+chatyuan_output_text+'\033[0m')
# since this model is multi-speaker and multi-lingual, we must set the target speaker and the language
# text to speech with a numpy output
# wav = tts_model.tts("This is a test! This is also a test!!", speaker=tts_model.speakers[0], language="en")
# text to speech to a file
if detected_language=='zh': # 'tts_models/zh-CN/baker/tacotron2-DDC-GST'
    tts_model.tts_to_file(text=chatyuan_output_text, file_path=output_audio_path)
elif detected_language=='en': # 'tts_models/multilingual/multi-dataset/your_tts'
    tts_model.tts_to_file(text=chatyuan_output_text, speaker=tts_model.speakers[0], language=tts_model.languages[0], file_path=output_audio_path)
print('--Successfully Generated Speech of Answer!')
print("--Finished")