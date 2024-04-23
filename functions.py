from ultralytics import YOLO
import cv2
import pandas as pd
import PIL
import pytesseract
import re
import spacy
import warnings
warnings.filterwarnings('ignore')
from PIL import Image


#load the pretrained YOLOv8n model
yolo_model = YOLO("models/yolo_model/train9/weights/best.pt") 

#load the pretrained spacy model
nlp = spacy.load("models/spacy_model/model-best")


def get_sign_stamps(image_path):
    # Empty list
    classes_present = []

    # Perform prediction using the preloaded model
    results = yolo_model.predict([image_path])
    result = results[0].boxes.cls.tolist()

    # Convert class indices to class names and append to the list
    for res in result:
      class_p = results[0].names[int(res)]
      if class_p == "mix":
        classes_present.append("stamp")
        classes_present.append("signature")
      else:
        classes_present.append(class_p)

    return classes_present  # Return the list of predicted classes


def ocr_ready(image_path, filename):

    # Read the original image
    original_image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    
    # new dimensions
    new_width = 1300
    new_height = 1400
    # Resize
    resized_image = cv2.resize(gray_image, (new_width, new_height))  

    # Apply Adaptive Histogram Equalization
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    equalized_image = clahe.apply(resized_image)
    
    filename = "ocr" + filename
    cv2.imwrite(filename, equalized_image)
    return filename

def extract(image_path):
    name = image_path.split('/')[-1]  # Extracting file name
    img_pl = PIL.Image.open(image_path)
    gray_image = img_pl.convert("L")
    data = pytesseract.image_to_data(gray_image)
    datalist = data.split('\n')
    data = []
    for line in datalist:
        line = line.split('\t')
        data.append(list(line)) 

    df = pd.DataFrame(data[1:], columns=data[0])
    df.dropna(inplace=True) #drop missing rows
    col_int = ["level", "page_num", "block_num", "par_num", "line_num", "word_num", "left", "top", "width", "height", "conf"]
    df[col_int] = df[col_int].astype(float)
    useful_data = df.query('conf >= 10.00')
    
    card = pd.DataFrame()
    card['text'] = useful_data['text']
    card['id'] = name
    
    # Drop rows containing empty spaces in the "text" column
    card = card[~card['text'].str.isspace()]

    # Reset index after dropping rows
    card.reset_index(drop=True, inplace=True)    
    
    temp = useful_data["text"]
    text = ""
    for i in temp:
        text = text + f"{str(i)} "

    # Replace multiple spaces with a single space
    cleaned_text = re.sub(r'\s+', ' ', text)
    return card, cleaned_text

def model(text):
  doc = nlp(text)
  name = str()
  dur = str()
  org = str()
  head = str()
  title = str()
  auth = str()
  
  for entity in doc.ents:
    parts = entity.label_.split("-")
    if len(parts) > 1:
      a = parts[1]
      if a == "NAME":
        name = name +" "+entity.text
      elif a == "ORG":
        org = org +" "+ entity.text
      elif a=="DUR":
        dur = dur +" "+ entity.text
      elif a=="TITLE":
        title = title +" "+ entity.text
      elif a=="HEAD":
        head = head +" "+ entity.text
      elif a=="AUTH":
        auth = auth+" "+entity.text
      
  data = {"name":name, "duration":dur, "organization":org, "supervisor":head, "title":title, "Authorizer":auth}
  return data


def convert_to_png(input_image_path):
    try:
        # Open the image file
        with Image.open(input_image_path) as img:
            # Extract the filename without the extension
            filename_without_extension = input_image_path.rsplit('.', 1)[0]

            # Convert the image to RGBA format (if it's not already)
            rgba_img = img.convert('RGBA')
            
            # Save the image in PNG format with the original filename
            output_image_path = f"{filename_without_extension}.png"
            rgba_img.save(output_image_path, 'PNG')
        
        return output_image_path
    
    except Exception as e:
        return f"Error occurred: {str(e)}"
      