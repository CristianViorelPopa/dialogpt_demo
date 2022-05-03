import streamlit as st
import os

import torch
from transformers import AutoModelWithLMHead, AutoTokenizer

import requests
import gdown


def download_file(url, local_filename):
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(local_filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)


def download_model_files(model_file_links, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    files_downloaded = True
    try:
        hostname = 'http://10.0.0.5:5555/'
        for _, filename in model_file_links:
            download_file(hostname + filename, os.path.join(output_dir, filename))
            print(f'Successfully downloaded {hostname + filename} to {os.path.join(output_dir, filename)}.')
    except Exception as e:
        print(e)
        print(f'Could not download {hostname + filename} to {os.path.join(output_dir, filename)}.')
        files_downloaded = False

    if files_downloaded:
        return

    files_downloaded = True
    try:
        for model_file_link, filename in model_file_links:
            gdown.download(model_file_link, os.path.join(output_dir, filename), quiet=False)
    except Exception as e:
        print(e)
        print(f'Could not download {model_file_link} to {os.path.join(output_dir, filename)}.')
        files_downloaded = False

    if files_downloaded:
        return


class Model:

    def __init__(self, model_file_links=[]):
        self.model_file_links = model_file_links
        self.model_directory = 'downloaded_nlp_model'
        self.model = None
        download_model_files(self.model_file_links, self.model_directory)

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_directory)
            self.model = AutoModelWithLMHead.from_pretrained(self.model_directory)
        except:
            print(f'Could not load DialoGPT tokenizer and/or model.')
            return

    # def __init__(self, s3_bucket, model_path):
    #     self.model_directory = 'downloaded_nlp_model'
    #
    #     if os.environ.get("DISABLE_NLP_MODEL") == "1":
    #         return
    #     if os.environ.get("DISABLE_NLP_MODEL_DOWNLOAD") != "1":
    #         download_s3_folder(s3_bucket, model_path, local_dir=self.model_directory)
    #
    #     self.tokenizer = AutoTokenizer.from_pretrained(self.model_directory)
    #     self.model = AutoModelWithLMHead.from_pretrained(self.model_directory)

    def answer(self, text, no_repeat_ngram_size=4, num_beams=1, top_k=50, top_p=0.8, temperature=0.8,
               num_candidate_answers=3):
        if self.model is None:
            return []

        new_user_input_ids = self.tokenizer.encode(text + self.tokenizer.eos_token, return_tensors='pt')
        chat_history_ids = torch.as_tensor([[]], dtype=torch.int32)
        bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1)
        output = self.model.generate(
            bot_input_ids, max_length=12000,
            pad_token_id=self.tokenizer.eos_token_id,
            no_repeat_ngram_size=no_repeat_ngram_size,
            do_sample=True,
            num_beams=num_beams,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            num_return_sequences=num_candidate_answers,
            return_dict_in_generate=True,
            output_scores=True
        )

        responses = []
        scores = []
        for idx in range(len(output.sequences)):
            responses.append(self.tokenizer.decode(output.sequences[:, bot_input_ids.shape[-1]:][idx],
                                                   skip_special_tokens=True))
            scores.append(torch.stack(list(output.scores), dim=0)[:, idx, :].max(dim=1).values.mean())
        return responses, scores


@st.cache(allow_output_mutation=True)
def setup_dialogpt():
    NLP_MODEL_FILE_LINKS = [
        ("https://drive.google.com/uc?id=1LNp86n3zQU3LynZrVJXu6r-v9X83rnXr", "pytorch_model.bin"),
        ("https://drive.google.com/uc?id=1g6FFoNisTWXvqPtK1kn_iXCHp3wgz8-u", "config.json"),
        ("https://drive.google.com/uc?id=12BJnluYylUmWLliGIe8P8I3Si_tM8pa_", "tokenizer_config.json"),
        ("https://drive.google.com/uc?id=1NEjKfWeZPiQ0Lwn641qQIeWFMXUAUKpP", "vocab.json"),
        ("https://drive.google.com/uc?id=1UFi3UDh7aCnNYTWQ232byzBSwZWJwaOS", "merges.txt"),
        ("https://drive.google.com/uc?id=1AK3nva0EJYfGWn3j-xKbSw79mEGHYIKi", "special_tokens_map.json"),
    ]
    model = Model(NLP_MODEL_FILE_LINKS)
    return model


st.title('DialoGPT Demo')

# initial setup
with st.spinner(text='In progress'):
    dialo_model = setup_dialogpt()

num_candidates = st.number_input('Number of candidate corrections', min_value=1, max_value=20, value=1,
                                 format='%d', help='DialoGPT is a generative model that may produce more than one '
                                                   'response for a given input')

st.write(
    '**Note**: The default values of the following parameters were the original values provided for DialoGPT generation, but you can tweak them for convenience.')

no_repeat_ngram_size = st.number_input('Size of n-grams that won\'t repeat', min_value=0, max_value=10, value=4,
                                       format='%d',
                                       help='If set to a number > 0, then n-grams (https://en.wikipedia.org/wiki/N-gram) of that size can only occur once.')

num_beams = st.number_input('Number of beams for text generation', min_value=1, max_value=10, value=1, format='%d',
                            help='Usually the more beams the better. Here\'s an article to understand the mechanisms of beam search: https://towardsdatascience.com/an-intuitive-explanation-of-beam-search-9b1d744e7a0f. Note that 1 beam means no beam search, so it\' a different way of generation. Here\'s another useful article: https://towardsdatascience.com/decoding-strategies-that-you-need-to-know-for-response-generation-ba95ee0faadc.')

top_k = st.slider('Top-k', min_value=10, max_value=100, value=50,
                  help='The model top-k parameter (read https://towardsdatascience.com/decoding-strategies-that-you-need-to-know-for-response-generation-ba95ee0faadc)')

top_p = st.slider('Top-p', min_value=0.5, max_value=1.0, value=0.8,
                  help='The model top-p parameter (read https://towardsdatascience.com/decoding-strategies-that-you-need-to-know-for-response-generation-ba95ee0faadc)')

temperature = st.slider('Temperature', min_value=0.3, max_value=2.0, value=0.8,
                        help='The model temperature parameter (read https://towardsdatascience.com/decoding-strategies-that-you-need-to-know-for-response-generation-ba95ee0faadc)')

# user form
with st.form(key='dialogpt'):
    input_text = st.text_input('Enter your text here:')
    generate_reply_submit = st.form_submit_button('Generate response')

    # on form submission
    if generate_reply_submit:
        responses, scores = dialo_model.answer(input_text,
                                               no_repeat_ngram_size=no_repeat_ngram_size,
                                               num_beams=num_beams,
                                               top_k=top_k,
                                               top_p=top_p,
                                               temperature=temperature,
                                               num_candidate_answers=num_candidates)
        st.write('This is what DialoGPT replied back:')
        output = ''
        for idx in range(len(responses)):
            output += f'{idx}. {responses[idx]} (score {scores[idx]})\n'
        st.write(output)
