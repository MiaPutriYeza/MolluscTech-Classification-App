# 🐚 MolluscTech - Native Clam Species Classifier

MolluscTech is an end-to-end web application developed to automatically identify and classify native clam species. This project effectively demonstrates the integration of deep learning models into a functional web environment to solve real-world regional identification challenges.

The system integrates two deep learning models: a lightweight **MobileNet** architecture to detect and filter out non-clam objects, and a custom **Convolutional Neural Network (CNN)** for specific species classification, ensuring highly efficient deployment and accurate results.

---

## ✨ Key Features

*   🧠 **Two-Stage Deep Learning Pipeline:** Utilizes MobileNetV2 as a gatekeeper to filter out irrelevant images, passing only valid images to the main classification model.
*   🎯 **High Accuracy (95%):** The custom CNN model achieves an impressive testing accuracy of 95% in distinguishing between clam species.
*   ⚡ **Real-Time Inference:** Powered by Flask to handle machine learning model inferences quickly and efficiently.
*   💻 **Modern & Responsive UI:** The frontend is crafted using HTML and Tailwind CSS to deliver a clean, seamless, and user-friendly experience across all devices.

## 🦪 Supported Species
The model is specifically trained to accurately classify five local clam species:
1.  **Kerang Bulu** (Hairy Clams)
2.  **Kerang Dara** (Blood Clams)
3.  **Kerang Batik** (Textile Clams)
4.  **Kerang Kepah** (Kepah Clams)
5.  **Kerang Hijau** (Green Mussels)

## 🛠️ Tech Stack

*   **Machine Learning:** TensorFlow, Keras (MobileNetV2), TensorFlow Lite (`.tflite`)
*   **Backend:** Python, Flask, Werkzeug
*   **Frontend:** HTML5, Tailwind CSS
*   **Data Processing:** NumPy, Pillow (PIL)
