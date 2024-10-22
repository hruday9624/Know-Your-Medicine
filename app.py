import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer

# App header
st.header("Know Your Medicine - Multiplication Table Generator")

# Load the model and tokenizer
@st.cache_resource
def load_model():
    model_name = "gpt2"  # Using GPT-2 for faster builds
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return model, tokenizer

# Load the model
model, tokenizer = load_model()

# Input for the number to generate the multiplication table
number = st.number_input("Enter a number:", min_value=1, max_value=100, value=5)

# Define the prompt for generating the multiplication table
prompt = f"Give me the multiplication table of {number} up to 12."

# Generate text based on the input
if st.button("Generate Multiplication Table"):
    # Tokenize the input prompt
    tokenized_input = tokenizer(prompt, return_tensors="pt")
    input_ids = tokenized_input["input_ids"]  # Using CPU for simplicity
    attention_mask = tokenized_input["attention_mask"]  # Using CPU for simplicity

    # Generate the response from the model
    response_token_ids = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_new_tokens=150,
        pad_token_id=tokenizer.eos_token_id
    )

    # Decode the generated tokens to text
    generated_text = tokenizer.decode(response_token_ids[0], skip_special_tokens=True)

    # Display the generated multiplication table
    st.write("Generated Multiplication Table:")
    st.write(generated_text)
