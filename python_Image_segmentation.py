import os

os.system("pip install torch torchvision opencv-python ultralytics easyocr streamlit")

import torch
import torchvision
from PIL import Image
# import matplotlib.pyplot as plt
from torchvision.transforms import functional as F
# import random
import cv2
import json
from ultralytics import YOLO
import pandas as pd
import easyocr
import streamlit as st
import shutil

IMAGE_PATH = 'market.jpg'
OUTPUT_PATH = 'outputs/'

model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
model.eval()

# Load an input image and convert it to a tensor
def segment_image(IMAGE_PATH):
    global model
    
    image = Image.open(IMAGE_PATH)
    image_tensor = F.to_tensor(image)

    # Add a batch dimension
    image_tensor = image_tensor.unsqueeze(0)

    # Perform the segmentation
    with torch.no_grad():
        predictions = model(image_tensor)



    # Extract the masks, boxes, and labels
    masks = predictions[0]['masks']
    boxes = predictions[0]['boxes']
    labels = predictions[0]['labels']

    # Plot the original image
    # plt.imshow(image)
    # plt.axis('off')

    # # Plot each object mask on the image
    # for i in range(len(masks)):
    #     mask = masks[i, 0].mul(255).byte().cpu().numpy()
    #     color = [random.randint(0, 255) for _ in range(3)]
    #     mask_image = Image.fromarray(mask)
    #     plt.imshow(mask_image, cmap='jet', alpha=0.5)

    # plt.show()


    # Create a directory to save the extracted objects
    os.makedirs(OUTPUT_PATH+'segmented_objects', exist_ok=True)

    # Loop through each mask and extract the object
    for i in range(len(masks)):
        mask = masks[i, 0].mul(255).byte().cpu().numpy()
        bbox = boxes[i].cpu().numpy().astype(int)

        # Crop the object using the bounding box
        x1, y1, x2, y2 = bbox
        cropped_image = image.crop((x1, y1, x2, y2))

        # Apply the mask to the cropped image
        mask_image = Image.fromarray(mask[y1:y2, x1:x2])
        cropped_image.putalpha(mask_image)

        # Save the cropped object with a unique ID
        object_id = f'object_{i}.png'
        cropped_image.save(os.path.join(OUTPUT_PATH+'segmented_objects', object_id))

        # Metadata (Optional)
        print(f"Object {i} saved with ID {object_id}")



    metadata = []

    # Save metadata for each object
    for i in range(len(masks)):
        bbox = boxes[i].cpu().numpy().astype(int).tolist()
        object_id = f'object_{i}.png'
        metadata.append({
            'object_id': object_id,
            'bbox': bbox
        })

    # Save metadata to a JSON file
    with open(OUTPUT_PATH+'segmented_objects/metadata.json', 'w') as f:
        json.dump(metadata, f, indent=4)



    # Load the pre-trained YOLOv5 model
    model = YOLO('yolov5su.pt')

    # Directory containing the segmented objects
    object_dir = OUTPUT_PATH+'segmented_objects'

    # Loop through each object image and identify it
    identifications = []
    for i in range(len(masks)):
        object_id = f'object_{i}.png'
        object_image_path = os.path.join(object_dir, object_id)

        # Perform object identification
        results = model(object_image_path)

        # Check if any detections were made
        if len(results[0].boxes) > 0:
            # Extract the best prediction (highest confidence)
            best_box = results[0].boxes[0]  # Assuming the first box is the best
            label = results[0].names[best_box.cls.item()]
            confidence = best_box.conf.item()

            # Store identification results
            identifications.append({
                'object_id': object_id,
                'label': label,
                'confidence': confidence
            })
        else:
            # Handle cases where no object was detected
            identifications.append({
                'object_id': object_id,
                'label': "No object detected",
                'confidence': 0
            })

    # Print out the identifications
    for ident in identifications:
        print(f"Object {ident['object_id']} identified as {ident['label']} with confidence {ident['confidence']:.2f}")



    # Initialize the EasyOCR reader
    reader = easyocr.Reader(['en'])

    # Directory containing the segmented objects
    object_dir = 'segmented_objects'

    # Loop through each object image and extract text
    extracted_texts = []
    for i in range(len(masks)):
        object_id = f'object_{i}.png'
        object_image_path = os.path.join(OUTPUT_PATH+object_dir, object_id)

        # Perform text extraction
        text_results = reader.readtext(object_image_path)

        # Combine the extracted text
        extracted_text = ' '.join([text[1] for text in text_results])

        # Store the extracted text
        extracted_texts.append({
            'object_id': object_id,
            'extracted_text': extracted_text
        })

    # Print out the extracted texts
    for text_data in extracted_texts:
        print(f"Text extracted from {text_data['object_id']}: {text_data['extracted_text']}")

    summaries = []

    # Generate summaries for each object
    for i in range(len(masks)):
        object_id = f'object_{i}.png'
        label = identifications[i]['label']
        extracted_text = extracted_texts[i]['extracted_text']

        # Create a summary
        summary = f"Object '{label}' identified with additional text: '{extracted_text}'."

        # Store the summary
        summaries.append({
            'object_id': object_id,
            'summary': summary
        })

    # Print out the summaries
    for summary in summaries:
        print(f"Summary for {summary['object_id']}: {summary['summary']}")



    # Mapping all data for each object
    data_mapping = []

    for i in range(len(masks)):
        object_id = f'object_{i}.png'
        label = identifications[i]['label']
        confidence = identifications[i]['confidence']
        extracted_text = extracted_texts[i]['extracted_text']
        summary = summaries[i]['summary']

        # Create a data structure
        data_mapping.append({
            'object_id': object_id,
            'label': label,
            'confidence': confidence,
            'extracted_text': extracted_text,
            'summary': summary,
            'bbox': metadata[i]['bbox']
        })

    # Save the mapping to a JSON file
    with open(OUTPUT_PATH+'segmented_objects/data_mapping.json', 'w') as f:
        json.dump(data_mapping, f, indent=4)

    # Print out the data mapping
    print(json.dumps(data_mapping, indent=4))


    # Load the original image
    annotated_image = cv2.imread(IMAGE_PATH)

    # Annotate the image with bounding boxes and labels
    for i in range(len(masks)):
        bbox = metadata[i]['bbox']
        label = identifications[i]['label']
        x1, y1, x2, y2 = bbox

        # Draw the bounding box
        cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (255, 0, 0), 2)

        # Put the label
        cv2.putText(annotated_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Save and display the annotated image
    cv2.imwrite(OUTPUT_PATH+'annotated_image.jpg', annotated_image)



    # Create a DataFrame from the data mapping
    df = pd.DataFrame(data_mapping)

    # Save the table as a CSV file
    df.to_csv(OUTPUT_PATH+'summary_table.csv', index=False)
    print("> ALL TASKS DONE")

processed_image_path = 'outputs/annotated_image.jpg'

if __name__ == '__main__':
    if os.path.exists('outputs'): shutil.rmtree('outputs')
    st.title('My Streamlit App')
    
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        input_image_path = 'input_image.jpg'
        with open(input_image_path, 'wb') as f:
            f.write(uploaded_file.getbuffer())
        
        # Open the saved image
        image = Image.open(input_image_path)
        
        # Display the uploaded image
        st.image(image, caption="Uploaded Image", use_column_width=True)

        st.text('Image Recieved, Processing .....')
        
        segment_image(input_image_path)
        
        # Display the processed image
        if os.path.exists(processed_image_path):
            st.image(processed_image_path, caption="Processed Image", use_column_width=True)
            
            with open('outputs/summary_table.csv', 'r') as file: 
                csv_data = file.read()
                st.download_button('Download CSV', csv_data, 'summary_table.csv', 'text/csv')
        else:
            st.error("Processed image not found.")