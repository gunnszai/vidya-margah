import os
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

def detect_dependencies(code):
    """
    Detect common ML libraries used in the code.

    Args:
        code: Python code string

    Returns:
        List of detected dependencies
    """

    dependencies = []

    common_imports = {
        'sklearn': 'scikit-learn',
        'pandas': 'pandas',
        'numpy': 'numpy',
        'matplotlib': 'matplotlib',
        'seaborn': 'seaborn',
        'tensorflow': 'tensorflow',
        'torch': 'torch',
        'cv2': 'opencv-python',
        'PIL': 'Pillow',
        'plotly': 'plotly',
        'keras': 'keras',
        'scipy': 'scipy',
        'xgboost': 'xgboost',
        'lightgbm': 'lightgbm'
    }

    for import_name, package_name in common_imports.items():
        if f"import {import_name}" in code or f"from {import_name}" in code:
            dependencies.append(package_name)

    return list(set(dependencies))  # Remove duplicates


def save_code_to_file(code, topic):
    """
    Save generated code to a file

    Args:
        code: Python code string
        topic: Topic name for filename

    Returns:
        Filename of saved code or None if failed
    """

    if not code or not code.strip():
        logger.warning("Cannot save empty code.")
        return None

    try:
        # Generate unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_topic = "".join(c if c.isalnum() else "_" for c in topic)[:30]
        filename = f"{safe_topic}_{timestamp}.py"
        filepath = os.path.join("generated_code", filename)

        # Create directory if it does not exist
        os.makedirs("generated_code", exist_ok=True)

        # Save code to file
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(code)

        logger.info(f"Code saved successfully: {filename}")
        return filename

    except Exception as e:
        logger.error(f"Failed to save code: {e}")
        return None