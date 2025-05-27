import os
import random
import datetime
from werkzeug.utils import secure_filename

def generate_custom_id():
    now = datetime.datetime.now()
    timestamp = now.strftime('%Y%m%d_%H%M%S')
    random_number = random.randint(1000, 9999)
    return f"IMG_{timestamp}_{random_number}"

def handle_image_upload(file):
    # Generate custom ID
    unique_id = generate_custom_id()

    # Get original extension
    _, ext = os.path.splitext(secure_filename(file.filename))
    ext = ext.lower()

    # New filename
    new_filename = f"{unique_id}{ext}"

    # Save path
    save_path = os.path.join('uploads', new_filename)

    # Ensure 'uploads/' folder exists
    os.makedirs('uploads', exist_ok=True)

    # Save file
    file.save(save_path)

    return new_filename
