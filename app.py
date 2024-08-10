import streamlit as st
import openai
import fal_client
import asyncio
import os
import time
import itertools
import requests
from PIL import Image
from io import BytesIO
from datetime import datetime

def tune_prompt_with_openai(prompt, model):
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY environment variable is not set")
    
    client = openai.OpenAI(api_key=openai_api_key)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You are an advanced AI assistant specialized in refining and enhancing image generation prompts. Your goal is to help users create more effective, detailed, and creative prompts for high-quality images. Respond with: 1) An improved prompt (prefix with 'PROMPT:'), 2) Explanation of changes (prefix with 'EXPLANATION:'), and 3) Additional suggestions (prefix with 'SUGGESTIONS:'). Each section should be on a new line."
            },
            {
                "role": "user",
                "content": f"Improve this image generation prompt: {prompt}"
            }
        ]
    )
    return response.choices[0].message.content.strip()

async def generate_image_with_fal(prompt, model, image_size, num_inference_steps, guidance_scale, num_images, safety_tolerance):
    fal_api_key = os.getenv("FAL_KEY")
    if not fal_api_key:
        raise ValueError("FAL_KEY environment variable is not set")
    
    os.environ['FAL_KEY'] = fal_api_key  # Set the API key as an environment variable
    
    handler = await fal_client.submit_async(
        model,
        arguments={
            "prompt": prompt,
            "image_size": image_size,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "num_images": num_images,
            "safety_tolerance": safety_tolerance
        }
    )

    return handler

def cycle_spinner_messages():
    messages = [
        "🎨 Mixing colors...",
        "✨ Sprinkling creativity dust...",
        "🖌️ Applying artistic strokes...",
        "🌈 Infusing with vibrant hues...",
        "🔍 Focusing on details...",
        "🖼️ Framing the masterpiece...",
        "🌟 Adding that special touch...",
        "🎭 Bringing characters to life...",
        "🏙️ Building the scene...",
        "🌅 Setting the perfect mood...",
    ]
    return itertools.cycle(messages)

def accept_tuned_prompt():
    st.session_state.user_prompt = st.session_state.tuned_prompt
    st.session_state.prompt_accepted = True

def save_image(url, prompt):
    response = requests.get(url)
    if response.status_code == 200:
        image = Image.open(BytesIO(response.content))
        
        # Create a filename using the current date and time
        timestamp = datetime.now().strftime("%Y-%m-%d-%H%M")
        
        # Sanitize the prompt to create a valid filename
        safe_prompt = "".join(c for c in prompt if c.isalnum() or c in (' ', '-', '_')).rstrip()
        safe_prompt = safe_prompt[:50]  # Limit the length of the prompt in the filename
        
        filename = f"{timestamp}_{safe_prompt}.png"
        
        # Ensure the images directory exists
        images_folder = os.path.join(os.path.dirname(__file__), 'images')
        os.makedirs(images_folder, exist_ok=True)
        
        # Save the image
        full_path = os.path.join(images_folder, filename)
        image.save(full_path)
        return full_path
    return None

def main():
    st.title("🤖 Image Generation with fal.ai & Flux")

    # Check for environment variables
    if not os.getenv("FAL_KEY"):
        st.error("FAL_KEY environment variable is not set. Please set it before running the app.")
        return

    # Model selection dropdown
    model_options = {
        "Flux Pro": "fal-ai/flux-pro",
        "Flux Dev": "fal-ai/flux/dev",
        "Flux Schnell": "fal-ai/flux/schnell",
        "Flux Realism": "fal-ai/flux-realism"
    }
    selected_model = st.selectbox("Select Model:", list(model_options.keys()), index=0)

    # Basic parameters
    image_size = st.selectbox("Image Size:", ["square_hd", "square", "portrait_4_3", "portrait_16_9", "landscape_4_3", "landscape_16_9"], index=0)
    num_inference_steps = st.slider("Number of Inference Steps:", min_value=1, max_value=50, value=28)

    # Advanced configuration in an expander
    with st.expander("Advanced Configuration", expanded=False):
        guidance_scale = st.slider("Guidance Scale:", min_value=1.0, max_value=20.0, value=3.5, step=0.1)
        safety_tolerance = st.selectbox("Safety Tolerance:", ["1", "2", "3", "4"], index=1)

    # Initialize session state
    if 'user_prompt' not in st.session_state:
        st.session_state.user_prompt = ""
    if 'tuned_prompt' not in st.session_state:
        st.session_state.tuned_prompt = ""
    if 'prompt_accepted' not in st.session_state:
        st.session_state.prompt_accepted = False

    # User input for the prompt
    user_prompt = st.text_input("Enter your image prompt:", value=st.session_state.user_prompt)

    # Update session state when user types in the input field
    if user_prompt != st.session_state.user_prompt:
        st.session_state.user_prompt = user_prompt
        st.session_state.prompt_accepted = False

    # OpenAI prompt tuning options
    use_openai_tuning = st.checkbox("Use OpenAI for prompt tuning", value=False)
    
    openai_model_options = ["gpt-4o", "gpt-4o-mini"]
    selected_openai_model = st.selectbox("Select OpenAI Model:", openai_model_options, index=0, disabled=not use_openai_tuning)

    if use_openai_tuning and user_prompt:
        if not os.getenv("OPENAI_API_KEY"):
            st.error("OPENAI_API_KEY environment variable is not set. Please set it before using OpenAI tuning.")
        else:
            if st.button("✏️ Tune Prompt"):
                with st.spinner("Tuning prompt with OpenAI..."):
                    try:
                        tuned_result = tune_prompt_with_openai(user_prompt, selected_openai_model)
                        
                        # Split the result into prompt, explanation, and suggestions
                        sections = tuned_result.split('\n')
                        for section in sections:
                            if section.startswith("PROMPT:"):
                                st.session_state.tuned_prompt = section.replace("PROMPT:", "").strip()
                            elif section.startswith("EXPLANATION:"):
                                explanation = section.replace("EXPLANATION:", "").strip()
                            elif section.startswith("SUGGESTIONS:"):
                                suggestions = section.replace("SUGGESTIONS:", "").strip()
                        
                        # Display the tuned prompt
                        st.subheader("Tuned Prompt:")
                        st.write(st.session_state.tuned_prompt)
                        
                        # Display explanation and suggestions in an expander
                        with st.expander("See explanation and suggestions"):
                            st.write("Explanation of changes:")
                            st.write(explanation)
                            st.write("Additional suggestions:")
                            st.write(suggestions)
                        
                        # Allow user to accept or regenerate the tuned prompt
                        col1, col2 = st.columns(2)
                        with col1:
                            st.button("✅ Accept Tuned Prompt", on_click=accept_tuned_prompt)
                        with col2:
                            if st.button("♻️ Regenerate Prompt"):
                                st.rerun()
                    except Exception as e:
                        st.error(f"Error tuning prompt: {str(e)}")

    if st.session_state.prompt_accepted:
        st.success("👍 Tuned prompt accepted and updated in the input field.")

    if st.button("☁️ Generate Image"):
        if not user_prompt:
            st.warning("⛔️ Please enter a prompt for image generation.")
            return

        # Display the prompt being used
        st.subheader("☁️ Generating image with the following prompt:")
        st.info(user_prompt)

        # Generate image with fal.ai
        try:
            spinner_placeholder = st.empty()
            message_cycle = cycle_spinner_messages()
            
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            # Create a task for image generation with num_images hardcoded to 1
            generation_task = loop.create_task(generate_image_with_fal(
                user_prompt, model_options[selected_model],
                image_size, num_inference_steps, guidance_scale, num_images=1, safety_tolerance=safety_tolerance
            ))

            # Update spinner message every 3 seconds until the task is complete
            while not generation_task.done():
                spinner_placeholder.text(next(message_cycle))
                try:
                    # Wait for 3 seconds or until the task completes, whichever comes first
                    loop.run_until_complete(asyncio.wait_for(asyncio.shield(generation_task), timeout=3))
                except asyncio.TimeoutError:
                    # This is expected every 3 seconds if the task isn't done
                    pass

            # Get the result
            handler = generation_task.result()
            result = loop.run_until_complete(handler.get())

            spinner_placeholder.empty()  # Clear the spinner

            # Display the generated image and save it
            st.subheader("🖼️ Your Generated Masterpiece:")
            image_info = result['images'][0]  # We know there's only one image
            st.image(image_info['url'], caption="Generated Image", use_column_width=True)
            
            # Save the image
            saved_path = save_image(image_info['url'], user_prompt)
            if saved_path:
                st.success(f"Image saved to {saved_path}")
            else:
                st.error("Failed to save the image")
            
            # Display additional information
            st.write(f"🌱 Seed: {result['seed']}")
            st.write(f"🚫 NSFW concepts detected: {result['has_nsfw_concepts']}")
            
        except Exception as e:
            st.error(f"⛔️ Error generating image: {str(e)}")

if __name__ == "__main__":
    main()