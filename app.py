from flask import Flask, render_template, request, jsonify, session, send_file
import os
import logging
from dotenv import load_dotenv
from werkzeug.utils import secure_filename
import secrets

# Import utility modules
from utility.genai_utils import call_genai
from utility.audio_utils import text_to_audio
from utility.code_executor import detect_dependencies, save_code_to_file
from utility.image_utils import generate_images, get_model_info

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', secrets.token_hex(16))

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create necessary directories
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs('generated_audio', exist_ok=True)
os.makedirs('generated_code', exist_ok=True)

# Read Gemini API key once at startup
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY', '')
HF_API_KEY = os.getenv('HF_API_KEY', '')


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# ─────────────────────────── Page Routes ───────────────────────────

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/text-explanation')
def text_explanation():
    return render_template('text_explaination.html')

@app.route('/code-generation')
def code_generation():
    return render_template('code_genration.html')

@app.route('/audio-learning')
def audio_learning():
    return render_template('audio_learning.html')

@app.route('/image-visualization')
def image_visualization():
    return render_template('image_visulization.html')

@app.route('/settings')
def settings():
    return render_template('settings.html')

@app.route('/about')
def about():
    return render_template('about.html')


# ─────────────────────────── API Routes ────────────────────────────

@app.route('/api/generate-text', methods=['POST'])
def generate_text():
    """Generate a text explanation using Gemini API."""
    try:
        data = request.get_json()
        topic = data.get('topic', '').strip()
        length = data.get('length', 'Brief')
        # Allow caller to override key; fall back to .env value
        api_key = data.get('api_key', GEMINI_API_KEY)

        if not topic:
            return jsonify({'error': 'Topic is required'}), 400
        if not api_key:
            return jsonify({'error': 'Gemini API key is not configured'}), 400

        result = call_genai(api_key, topic, length, "Text explanation")
        if result:
            briefing, _, _, _ = result
            return jsonify({'success': True, 'content': briefing})
        return jsonify({'success': False, 'error': 'Failed to generate content'}), 500

    except Exception as e:
        logger.error(f"Error in generate_text: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/generate-code', methods=['POST'])
def generate_code():
    """Generate ML code + explanation using Gemini API."""
    try:
        data = request.get_json()
        topic = data.get('topic', '').strip()
        length = data.get('length', 'Detailed')
        api_key = data.get('api_key', GEMINI_API_KEY)

        if not topic:
            return jsonify({'error': 'Topic is required'}), 400
        if not api_key:
            return jsonify({'error': 'Gemini API key is not configured'}), 400

        result = call_genai(api_key, topic, length, "Code with explanation")
        if result:
            briefing, code_content, _, _ = result
            dependencies = detect_dependencies(code_content) if code_content else []
            code_file = None
            if code_content:
                code_file = save_code_to_file(code_content, topic)
            return jsonify({
                'success': True,
                'explanation': briefing,
                'code': code_content,
                'dependencies': dependencies,
                'code_file': code_file
            })
        return jsonify({'success': False, 'error': 'Failed to generate code'}), 500

    except Exception as e:
        logger.error(f"Error in generate_code: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/generate-audio', methods=['POST'])
def generate_audio():
    """Generate an audio explanation using Gemini API + gTTS."""
    try:
        data = request.get_json()
        topic = data.get('topic', '').strip()
        length = data.get('length', 'Brief')
        api_key = data.get('api_key', GEMINI_API_KEY)

        if not topic:
            return jsonify({'error': 'Topic is required'}), 400
        if not api_key:
            return jsonify({'error': 'Gemini API key is not configured'}), 400

        result = call_genai(api_key, topic, length, "Audio")
        if result:
            briefing, _, audio_script, _ = result
            # Use audio script if generated, otherwise use the full briefing
            text_for_audio = audio_script if audio_script else briefing
            audio_file = text_to_audio(text_for_audio, topic)
            if audio_file:
                return jsonify({
                    'success': True,
                    'content': briefing,
                    'audio_file': audio_file
                })
            return jsonify({'success': False, 'error': 'Audio generation failed'}), 500
        return jsonify({'success': False, 'error': 'Failed to generate content'}), 500

    except Exception as e:
        logger.error(f"Error in generate_audio: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/generate-images', methods=['POST'])
def generate_images_route():
    """Generate educational images using Gemini API."""
    try:
        data = request.get_json()
        topic = data.get('topic', '').strip()
        length = data.get('length', 'Brief')
        backend = data.get('backend', 'Google Gemini (Fast & Free)')
        api_key = data.get('api_key', GEMINI_API_KEY)
        hf_key = data.get('hf_key', HF_API_KEY)

        if not topic:
            return jsonify({'error': 'Topic is required'}), 400
        if not api_key:
            return jsonify({'error': 'Gemini API key is not configured'}), 400

        # First get explanation + image prompts from Gemini
        result = call_genai(api_key, topic, length, "Image Explanation")
        if not result:
            return jsonify({'success': False, 'error': 'Failed to generate image prompts'}), 500

        briefing, _, _, image_prompts = result

        # Fall back to a generic prompt if none were extracted
        if not image_prompts:
            image_prompts = [f"Educational diagram explaining {topic} in machine learning"]

        images = generate_images(image_prompts, api_key, hf_key, backend)
        return jsonify({
            'success': True,
            'content': briefing,
            'images': images,
            'prompts': image_prompts
        })

    except Exception as e:
        logger.error(f"Error in generate_images_route: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/audio/<filename>')
def serve_audio(filename):
    """Serve a generated audio file."""
    try:
        filepath = os.path.join('generated_audio', secure_filename(filename))
        if not os.path.exists(filepath):
            return jsonify({'error': 'Audio file not found'}), 404
        return send_file(filepath, mimetype='audio/mpeg')
    except Exception as e:
        logger.error(f"Error serving audio: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/model-info')
def model_info():
    """Return information about the AI models in use."""
    return jsonify(get_model_info())


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)