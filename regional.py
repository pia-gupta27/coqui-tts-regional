
import os
import torch  
import TTS
from TTS.tts.configs.shared_configs import BaseDatasetConfig, CharactersConfig
from TTS.tts.configs.vits_config import VitsConfig
from TTS.utils.audio import AudioProcessor
from TTS.tts.models.vits import Vits
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.tts.datasets import load_tts_samples
from TTS.utils.synthesizer import Synthesizer
from TTS.tts.utils.text.phonemizers import ESpeak



class TextToSpeech:
    def __init__(self):
        # Initialize the model configurations
        self.model_path = "vits_model.pth"
        self.config_path = "vits_config.json"
        self.output_dir = "output_audio"


        character_config = CharactersConfig(
            characters_class="TTS.tts.models.vits.VitsCharacters",
            pad="<PAD>",
            eos="<EOS>",
            bos="<BOS>",
            blank="<BLNK>",
            phonemes = None,
            characters = "ABCDEFGHIJKLMNOPRSTVWXYZabcdefghijklmnopqrstuvwxyzँगऊोग़डटणढ़ॉएपदझ़ंृघभसछिठक़कःहऔजाओत्ऋऐधईीथञज़लूखढचऑबनवशफआयख़ौड़रइऍअमफ़ॠैउषुेँंः",
            punctuations = "|।–!,-. ?"
        )

        # Initialize dataset and model configuration
        self.config = VitsConfig(
            batch_size=32,
            eval_batch_size=16,
            epochs=100,
            text_cleaner="basic_cleaners",
            use_phonemes=True,
            phoneme_language="hi",  # Set Hindi as the phoneme language
            characters=character_config,
            datasets=[BaseDatasetConfig(formatter="mozilla", path="@/TTS/TTS/tts/datasets/common_voice_hindi")],
        )
        
        # Initialize AudioProcessor
        self.ap = AudioProcessor.init_from_config(self.config)

        # Initialize Tokenizer
        phonemizer = ESpeak(language="hi")
        self.tokenizer, self.config = TTSTokenizer.init_from_config(self.config, phonemizer=phonemizer)


        # Initialize VITS model
        self.model = Vits(self.config, self.ap, self.tokenizer)

        # Load pre-trained model checkpoint (replace with your trained model path)
        checkpoint = torch.load(self.model_path, map_location="cpu")
        self.model.load_state_dict(checkpoint["model"])
        self.model.eval()

        # Synthesizer for generating speech
        self.synthesizer = Synthesizer(self.model, self.ap, self.config)

        # Create output directory if not exists
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def synthesize_speech(self, text):
        # Tokenize the input text
        inputs = self.tokenizer.text_to_sequence(text)

        # Generate speech (convert text to audio)
        wav = self.synthesizer.tts(inputs)
        
        # Save the output WAV file
        output_wav_path = os.path.join(self.output_dir, "output.wav")
        self.ap.save_wav(wav, output_wav_path)

        return output_wav_path

