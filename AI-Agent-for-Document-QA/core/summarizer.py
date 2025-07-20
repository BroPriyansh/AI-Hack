from transformers import pipeline

def summarize_markdown(markdown_text, api_key=None, model="sshleifer/distilbart-cnn-12-6"):
    summarizer = pipeline("summarization", model=model)
    
    max_input_length = 1024
    if len(markdown_text) > max_input_length:
        markdown_text = markdown_text[:max_input_length]

    summary = summarizer(markdown_text, max_length=150, min_length=40, do_sample=False)
    return summary[0]['summary_text']
