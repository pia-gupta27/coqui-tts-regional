import streamlit as st
from regional import TextToSpeech

# Set up the Streamlit app
st.title("Hindi Text-to-Speech (TTS) Application")
st.write("This app allows you to convert Hindi text into speech using Mozilla Common Voice dataset and VITS model.")

# Input text from the user
hindi_text = st.text_area("Enter Hindi Text:", placeholder="अपना हिंदी टेक्स्ट यहाँ डालें")

# Button to trigger TTS
if st.button("Convert to Speech"):
    if hindi_text:
        # Initialize TTS object
        regional = TextToSpeech()
        # Convert text to speech
        output_wav_path = regional.synthesize_speech(hindi_text)

        # Play the generated speech
        audio_file = open(output_wav_path, 'rb')
        audio_bytes = audio_file.read()
        st.audio(audio_bytes, format='audio/wav')
        
        st.success("Speech generated successfully!")
    else:
        st.warning("Please enter some Hindi text to convert.")

# Footer
st.markdown("Created by Pia Gupta using Coqui TTS and Mozilla Common Voice Dataset")
