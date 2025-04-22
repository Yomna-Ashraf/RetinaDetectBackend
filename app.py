from flask import Flask, request, jsonify
from transformers import AutoModelForImageClassification, AutoProcessor
from PIL import Image
import io
import fitz  # PyMuPDF
from flask_cors import CORS, cross_origin

app = Flask(__name__)
CORS(app, origins=["http://localhost:5173", "https://your-frontend.onrender.com"])

model_name = "AsmaaElnagger/Diabetic_RetinoPathy_detection"
model = AutoModelForImageClassification.from_pretrained(model_name)
processor = AutoProcessor.from_pretrained(model_name)

def pdf_to_images_pymupdf(pdf_data):
    try:
        pdf_document = fitz.open(stream=pdf_data, filetype="pdf")
        images = []
        for page_num in range(pdf_document.page_count):
            page = pdf_document.load_page(page_num)
            pix = page.get_pixmap()
            img_data = pix.tobytes("jpeg")  # Or "png"
            images.append(img_data)
        return images
    except Exception as e:
        print(f"Error converting PDF: {e}")
        return None

@app.route('/classify', methods=['POST'])
def classify_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    uploaded_file = request.files['file']
    file_type = uploaded_file.filename.rsplit('.', 1)[1].lower()

    try:
        if file_type in ['jpg', 'jpeg', 'png', 'gif']:
            # Handle image upload
            img_data = uploaded_file.read()
            image = Image.open(io.BytesIO(img_data)).convert("RGB")
            inputs = processor(images=image, return_tensors="pt")
            outputs = model(**inputs)
            logits = outputs.logits
            predicted_class_idx = logits.argmax(-1).item()
            result = model.config.id2label[predicted_class_idx]
            return jsonify({'result': result})

        elif file_type == 'pdf':
            # Handle PDF upload
            pdf_data = uploaded_file.read()
            images = pdf_to_images_pymupdf(pdf_data)

            if images:
                # Process the first image in the pdf, you may need to loop through all images.
                image = Image.open(io.BytesIO(images[0])).convert("RGB")
                inputs = processor(images=image, return_tensors="pt")
                outputs = model(**inputs)
                logits = outputs.logits
                predicted_class_idx = logits.argmax(-1).item()
                result = model.config.id2label[predicted_class_idx]
                return jsonify({'result': result})
            else:
                return jsonify({'error': 'PDF conversion failed.'}), 500

        else:
            return jsonify({'error': 'Unsupported file type'}), 400

    except Exception as e:
        return jsonify({'error': f'An error occurred: {e}'}), 500

if __name__ == '__main__':
    app.run(port=5001, debug=True)