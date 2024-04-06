SAMPLING_RATE = 16000

import torch
torch.set_num_threads(1)

from IPython.display import Audio
from pprint import pprint
# download example
#torch.hub.download_url_to_file('https://models.silero.ai/vad_models/en.wav', 'en_example.wav')
USE_ONNX = False # change this to True if you want to test onnx model
  
model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                              model='silero_vad',
                              force_reload=True,
                              onnx=USE_ONNX)

(get_speech_timestamps,
 save_audio,
 read_audio,
 VADIterator,
 collect_chunks) = utils

input_file = '20221025.wav'
output_file = '20221025_vad.wav'

#input_file = '20230606.wav'
#output_file = '20230606_vad.wav'

#input_file = '20230829.wav'
#output_file = '20230829_vad.wav'

#input_file = '20231225.wav'
#output_file = '20231225_vad.wav'

#input_file = '20240130.wav'
#output_file = '20240130_vad.wav'

wav = read_audio(input_file, sampling_rate=SAMPLING_RATE)
# get speech timestamps from full audio file
speech_timestamps = get_speech_timestamps(wav, model, sampling_rate=SAMPLING_RATE)
pprint(speech_timestamps)

# merge all speech chunks to one audio
save_audio(output_file,
           collect_chunks(speech_timestamps, wav), sampling_rate=SAMPLING_RATE) 
Audio(output_file)

## using VADIterator class

vad_iterator = VADIterator(model)
wav = read_audio(input_file, sampling_rate=SAMPLING_RATE)

window_size_samples = 1536 # number of samples in a single audio chunk
for i in range(0, len(wav), window_size_samples):
    chunk = wav[i: i+ window_size_samples]
    if len(chunk) < window_size_samples:
      break
    speech_dict = vad_iterator(chunk, return_seconds=True)
    if speech_dict:
        print(speech_dict, end=' ')
vad_iterator.reset_states() # reset model states after each audio

## just probabilities

wav = read_audio(input_file, sampling_rate=SAMPLING_RATE)
speech_probs = []
window_size_samples = 1536
for i in range(0, len(wav), window_size_samples):
    chunk = wav[i: i+ window_size_samples]
    if len(chunk) < window_size_samples:
      break
    speech_prob = model(chunk, SAMPLING_RATE).item()
    speech_probs.append(speech_prob)
vad_iterator.reset_states() # reset model states after each audio

print(speech_probs[:10]) # first 10 chunks predicts


