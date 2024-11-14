import spacy
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Setup spaCy NLP
nlp = None

# Setup OpenAI API
openai_client = None

# Setup Claude 2 API
bedrock = None
anthropic_client = None


def sentencize(text):
    """Split text into sentences"""
    global nlp
    if not nlp:
        nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    return [sent for sent in doc.sents]


def split_text(text, segment_len=200):
    """Split text into segments according to sentence boundaries."""
    segments, seg = [], []
    sents = [[token.text for token in sent] for sent in sentencize(text)]
    for sent in sents:
        if len(seg) + len(sent) > segment_len:
            segments.append(" ".join(seg))
            seg = sent
            # single sentence longer than segment_len
            if len(seg) > segment_len:
                # split into chunks of segment_len
                seg = [
                    " ".join(seg[i:i+segment_len])
                    for i in range(0, len(seg), segment_len)
                ]
                segments.extend(seg)
                seg = []
        else:
            seg.extend(sent)
    if seg:
        segments.append(" ".join(seg))
    return segments


def get_model_batch_response(
    prompts: list[str],
    model: str,
    temperature=0,
    n_choices=1, 
    max_new_tokens=500,
    **kwargs
):
    """
    Get batch generation results with given prompts using HuggingFace Transformers.

    Parameters
    ----------
    prompts : List[str]
        List of prompts for generation.
    temperature : float, optional
        The generation temperature, use greedy decoding when setting
        temperature=0, defaults to 0.
    model : str, optional
        The model name from HuggingFace Hub, defaults to 'gpt2'.
    n_choices : int, optional
        How many samples to return for each prompt input, defaults to 1.
    max_new_tokens : int, optional
        Maximum number of newly generated tokens, defaults to 500.

    Returns
    -------
    response_list : List[str]
        List of generated text.
    """
    if not prompts or len(prompts) == 0:
        raise ValueError("Invalid input.")

    tokenizer = AutoTokenizer.from_pretrained(model)
    model = AutoModelForCausalLM.from_pretrained(model)
    
    if torch.cuda.is_available():
        model = model.to('cuda')

    response_list = []
    
    for prompt in prompts:
        if len(prompt) == 0:
            raise ValueError("Invalid prompt.")
            
        # Handle both string and message list formats
        if isinstance(prompt, str):
            input_text = prompt
        elif isinstance(prompt, list):
            # Concatenate messages into a single string
            input_text = " ".join([m['content'] for m in prompt])
        else:
            raise ValueError("Invalid prompt type.")

        # Tokenize input
        inputs = tokenizer(input_text, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = inputs.to('cuda')

        # Generate multiple choices if requested
        outputs = []
        for _ in range(n_choices):
            with torch.inference_mode():
                output = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature if temperature > 0 else 1.0,
                    do_sample=temperature > 0,
                    num_return_sequences=1,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            # Decode and clean up the generated text
            generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
            # Remove the input prompt from the generated text
            response = generated_text[len(input_text):].strip()
            outputs.append(response)
        
        if n_choices == 1:
            response_list.append(outputs[0])
        else:
            response_list.append(outputs)

    return response_list