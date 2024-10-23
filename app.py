import streamlit as st
import google.generativeai as genai


# App header
st.header("Know Your Medicine")

# Retrieve the API key from Streamlit secrets
GOOGLE_API_KEY = st.secrets["GEMINI_API_KEY"]

# Configure the Google Generative AI API with your API key
genai.configure(api_key=GOOGLE_API_KEY)

# Input field for the medicine name
st.subheader("Enter Medicine Details:")
medicine_name = st.text_input('Medicine Name', '')

# Create the prompt based on user input
if medicine_name:
    prompt = f"""
    Analyze the following details:
    1. Write the medicine name and purpose of the medicine.
    2. Write down the symptoms for which this medicine should be used.
    3. List the possible side effects of the medicine.
    4. Mention any common drug interactions or contraindications.
    5. Provide common brand names or generic alternatives, if available.
    6. Mention any specific precautions (e.g., avoid alcohol, potential allergies).

    Medicine Name = {medicine_name}
    """

# Button to submit the prompt
if st.button("Generate"):
    if medicine_name:  # Ensure the medicine name is entered
        try:
            # Initialize the generative model (adjust model name if needed)
            model = genai.GenerativeModel('gemini-pro')  # Ensure this is the correct model name

            # Generate content based on the prompt
            response = model.generate_content(prompt)
            
            # Check if there is a response from the model
            if response:
                st.subheader("Generated Medicine Analysis:")
                st.write(response.text)  # Display the generated response
            else:
                st.error("Error: Unable to generate the analysis.")
                
        except Exception as e:
            st.error(f"Error: {e}")
    else:
        st.error("Please enter a medicine name to generate the analysis.")

# Add space or content at the bottom
st.write("\n" * 20)  # Adds space to push the content down

# Footer
st.markdown("Built with 🧠 by Hruday & Google Gemini")
