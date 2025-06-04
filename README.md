# 🛒 Automated Grocery Checkout System

[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.0+-red.svg)](https://streamlit.io/)
[![YOLO](https://img.shields.io/badge/YOLO-v8-yellow.svg)](https://github.com/ultralytics/yolov5)

A modern, AI-powered grocery checkout system that uses computer vision to automatically detect and bill items in real-time. Perfect for retail stores looking to modernize their checkout process and reduce wait times.

## 🎥 Demo


https://github.com/user-attachments/assets/a1826305-74ac-4127-80e9-2307c96ec50c

> Note: The demo video shows slower processing due to CPU-only execution. Performance significantly improves with GPU acceleration.

## ✨ Features

- **Real-time Object Detection**: Uses YOLO (You Only Look Once) model for fast and accurate product detection
- **Automated Billing**: Automatically generates bills based on detected items
- **User-friendly Interface**: Clean and intuitive web interface built with Streamlit
- **Smart Tracking**: Prevents duplicate item scanning with cooldown system
- **Performance Metrics**: Real-time FPS monitoring and detection statistics
- **PDF Bill Generation**: Creates professional PDF bills for transactions

## 🚀 Getting Started

### Prerequisites

- Python 3.7 or higher
- Webcam or video input device

### Installation

1. Clone the repository:
```bash
git clone   git clone https://github.com/SripadhiDeekshith/GroceryCheckoutSystem.git

cd GroceryCheckoutSystem
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
streamlit run app.py
```

## 💡 How It Works

1. The system uses your camera feed to detect products in real-time
2. When a product is detected, it's added to the bill after a cooldown period to prevent duplicate scanning
3. The total is automatically calculated and displayed
4. Generate a PDF bill when the transaction is complete

## 🛠️ Built With

- [Streamlit](https://streamlit.io/) - The web framework used
- [YOLOv8](https://github.com/ultralytics/yolov5) - Object detection model
- [OpenCV](https://opencv.org/) - Computer vision tasks
- [FPDF](https://pyfpdf.readthedocs.io/en/latest/) - PDF generation

## 🤝 Contributing

Contributions are welcome! Feel free to submit a Pull Request.

## 📧 Contact

If you have any questions or suggestions, feel free to reach out!

---
