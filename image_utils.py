import requests
import base64
from google import genai
from google.genai import types
import logging
from PIL import Image
from io import BytesIO

logger = logging.getLogger(__name__)


def generate_images(prompts, gemini_key, hf_key, backend):
    """
    Generate images using either Google Gemini or Hugging Face backend

    Args:
        prompts: List of image prompts
        gemini_key: Google Gemini API key
        hf_key: HuggingFace API key
        backend: Backend to use for generation

    Returns:
        List of base64-encoded image URLs
    """

    if backend == "Google Gemini (Fast & Free)":
        return _gen_with_gemini(prompts, gemini_key)
    else:
        return _gen_with_hf(prompts, hf_key)


def _gen_with_gemini(prompts, api_key):
    """Generate images using Gemini 2.0 Flash Preview Image Generation model"""
    try:
        # Configure the client
        client = genai.Client(api_key=api_key)

        urls = []

        for i, prompt in enumerate(prompts):
            try:
                # Enhance prompt for better educational content
                enhanced_prompt = _enhance_educational_prompt(prompt)

                logger.info(f"Generating image {i+1}/{len(prompts)} with Gemini...")

                # Use Gemini 2.0 Flash Preview Image Generation
                response = client.models.generate_content(
                    model="gemini-2.0-flash-preview-image-generation",
                    contents=f"Generate an image: {enhanced_prompt}",
                    config=types.GenerateContentConfig(
                        response_modalities=['TEXT', 'IMAGE']
                    )
                )

                # Extract image from response
                for part in response.candidates[0].content.parts:
                    if part.inline_data is not None:
                        img_bytes = part.inline_data.data
                        b64 = base64.b64encode(img_bytes).decode()
                        urls.append(f"data:image/png;base64,{b64}")

                        logger.info(f"Successfully generated image {i+1}")
                        break

            except Exception as e:
                logger.error(f"Error generating image {i+1}: {str(e)}")
                continue

        return urls

    except Exception as e:
        logger.error(f"Gemini image generation failed: {str(e)}")
        return []


def _enhance_educational_prompt(original_prompt):
    """
    Enhance prompts for better educational visual content

    Args:
        original_prompt: Original image prompt

    Returns:
        Enhanced prompt with educational styling
    """

    # Add educational visual style elements
    enhancements = [
        "educational diagram style",
        "clean and professional",
        "technical illustration",
        "white background",
        "high contrast",
        "clear and readable"
    ]

    if not any(word in original_prompt.lower() for word in ["style", "background", "diagram", "illustration"]):
        enhanced = f"{original_prompt}, {', '.join(enhancements)}"
    else:
        enhanced = original_prompt

    return enhanced


def get_model_info():
    """
    Return information about the current models

    Returns:
        Dictionary with model information
    """

    return {
        "gemini_model": "gemini-2.0-flash-preview-image-generation",
        "gemini_fallback": "imagen-3.0-generate-002",
        "hf_model": "stabilityai/stable-diffusion-xl-base-1.0",

        "hf_info": {
            "parameters": "3.5B",
            "architecture": "Latent Diffusion Model",
            "features": [
                "High-resolution generation",
                "Better prompt adherence",
                "Enhanced image quality"
            ],
            "native_resolution": "1024x1024",
            "recommended_steps": "20-28"
        }
    }