from functions import *
from flask import Flask, jsonify, request
from pathlib import Path
from werkzeug.utils import secure_filename
import sys

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore") 
    
app = Flask(__name__)

@app.route("/test")
def test():
    return jsonify({"Message": "Hello User, API is Working."})
 
@app.route("/certificates", methods=["POST"])
def certificates():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file part"}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400

        # Save the uploaded file with its original filename
        filename = secure_filename(file.filename)
        file.save(f'data/{filename}')
        
        img = convert_to_png(f'data/{filename}')
        
        try:
            ocr_img = ocr_ready(img, filename)
            text = extract(ocr_img)[1]
            # Delete the OCR image after use
            Path(ocr_img).unlink()
            ner = model(text) #ERRORRRRRRRRR HEEEERRRRRREEEEEEEEEEEEEEEEEEEEEEEEE
        except Exception as e:
            return jsonify({"error": f"Error processing OCR: {str(e)}"}), 500
    
        try:
            yolo_output = get_sign_stamps(img)
        except Exception as e:
            return jsonify({"error": f"Error performing sign and stamp detection: {str(e)}"}), 500

        details = {
            "file": filename,
            "certificate": text,
            "signatures and stamps": yolo_output,
            "NER": ner
        }

        return jsonify({"verify": details}), 200
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
