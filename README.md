🛠️ Submersible Pump Impeller Defect Detection & Analysis
AI-powered Industrial Quality Inspection (Computer Vision + Reasoning)

This project is a complete automated visual inspection system built to detect manufacturing defects in submersible pump impellers and provide brief diagnostic analysis.
It combines object detection, visual reasoning, and batch processing to deliver a practical Industry-4.0–ready inspection workflow.

🚀 Key Features

Accurate defect detection using a custom-trained object detection model

Automatic defect localization with bounding boxes

AI-generated analysis describing:

What the defect is

Why it likely occurred

Suggested repair or corrective action

Supports both single-image and batch-image inspection

Drag-and-drop interface built with Gradio

GPU-accelerated inference for fast real-time processing

Outputs annotated images + text reports for each component

🔧 Tech Stack

Python

YOLO-based object detection

Vision-language reasoning (LLM-based analysis)

Gradio UI

CUDA GPU acceleration

Pillow, Torch, Ultralytics, Ollama API

📁 Project Structure
project/
│
├── main.py                     # Core detection + analysis pipeline
├── ui.py                       # Gradio user interface
├── Finetuned Model.pt          # Custom trained model (ignored on GitHub)
├── inspector_outputs/          # Output reports + annotated images
├── requirements.txt
└── README.md                   # Documentation

🖼️ How It Works
1️⃣ Upload image(s)

Drag-and-drop a single impeller image or a folder.

2️⃣ Defect Detection

The YOLO model identifies defect regions and extracts them.

3️⃣ Visual Reasoning

A lightweight vision-language model analyzes each defect.

4️⃣ Output

For each image, the system generates:

Annotated image (full.jpg)

Inspection report (inspection.txt)

Individual defect crops

🖥️ User Interface (Gradio)

The UI allows:

✔ Selecting single image
✔ Selecting folder of images
✔ Viewing annotated results
✔ Reading AI-generated defect insights

⚙️ How to Run Locally
1️⃣ Install dependencies
pip install -r requirements.txt

2️⃣ Start your Ollama server
ollama serve

3️⃣ Run the Gradio interface
python ui.py


A local URL will appear — open it in your browser.

📊 Example Output

Annotated Image
(Add this screenshot later)

![Annotated](screenshots/annotated_example.jpg)


Generated Report

defect: surface crack near blade edge  
cause: likely due to casting stress or uneven cooling  
repair: smoothing or reworking the affected region recommended  

🏭 Why This Project Matters

This system demonstrates how AI can transform industrial quality control by:

Reducing manual inspection labor

Providing consistent, repeatable defect assessment

Improving production line throughput

Supporting early detection during casting & machining

Enabling Industry 4.0 automation workflows
