ML-Based VPN-Aware Intrusion Detection System

A Machine Learning-based cybersecurity application that combines VPN Traffic Detection and Network Intrusion Detection (IDS) within a single framework. The system analyzes network flow data, identifies VPN/Non-VPN traffic, classifies multiple cyberattacks, monitors concept drift, and presents results through an interactive Streamlit dashboard.

Project Overview

Traditional Intrusion Detection Systems (IDS) primarily focus on detecting attacks in normal network traffic, often overlooking malicious activities hidden within encrypted VPN communication.

This project addresses that challenge by integrating:

   VPN Traffic Detection
   Network Attack Classification
   Concept Drift Monitoring
   Interactive Streamlit Dashboard

into one unified cybersecurity solution.



 Features

  Detects VPN and Non-VPN network traffic
  Classifies multiple cyberattack categories
  Supports CSV file upload for prediction
  Interactive Streamlit dashboard
  Attack analytics and visualization
  Threat level and risk score calculation
  Concept drift detection using Jensen-Shannon Distance
  Machine Learning-based prediction


Machine Learning Models

The following models were implemented and evaluated:

 Random Forest
 Multi-Layer Perceptron (MLP)
 XGBoost

Best Model Performance

Random Forest achieved the best overall performance and was selected for deployment in the application.



Datasets

This project uses two publicly available cybersecurity datasets.

UNSW-NB15 Dataset

Purpose: Network Intrusion Detection

Records: ~2,059,417 network traffic records

Attack Categories

Normal
Analysis
Backdoors
DoS
Exploits
Fuzzers
Generic
Reconnaissance
Shellcode
Worms

The dataset was used to train machine learning models for multi-class attack classification.


ISCX VPN-NonVPN Dataset

Purpose: VPN Traffic Detection

Records: ~142,963 network traffic records

Classes

VPN Traffic
Non-VPN Traffic

The dataset was used to identify encrypted VPN communication using network flow features without inspecting packet payloads.



Dataset Preparation

The datasets were preprocessed before model training.

Processing steps included:

 Merging multiple CSV files
 Removing missing values
 Handling inconsistent data
 Feature selection
 One-Hot Encoding
 Label Encoding
 Data preprocessing for machine learning

The final processed datasets were:

vpn_merged.csv
unsw_merged_cleaned.csv



 Note
 Due to GitHub file size limitations, the original datasets are not included in this repository. Only sample input files are provided for testing.



Project Structure


vpn_aware_ids/
 app/
 app.py
 scripts/
 data/
 uploads/
.streamlit/
 requirements.txt
.gitignore
 README.md




 Installation

Clone the repository

bash
git clone https://github.com/Pragati2004T/ML--Based--VPN-Detection-.git


Go to project directory

bash
cd ML--Based--VPN-Detection

Install dependencies

bash
pip install -r requirements.txt


Run the Streamlit application

bash
streamlit run app/app.py


Input

The application accepts:

CSV files
Excel files

containing network traffic features.



Output

The system predicts:

VPN / Non-VPN Traffic
Attack Type
Prediction Confidence
Threat Level
Risk Score
Attack Analytics
Concept Drift Status



Dashboard Features


  VPN Traffic Distribution
  Attack Category Distribution
  Threat Level Indicator
  Risk Score
  Prediction Summary
  Concept Drift Detection
  Attack Intelligence Report



Application Screenshots

<img width="552" height="323" alt="image" src="https://github.com/user-attachments/assets/7e25dad1-8c9c-41b4-9b12-794843ac0532" />

<img width="530" height="241" alt="image" src="https://github.com/user-attachments/assets/731b40be-e712-486c-af3d-8e7b920bbfba" />

<img width="459" height="260" alt="image" src="https://github.com/user-attachments/assets/65f1a39f-035a-4aa2-a2b4-6f0e791e1f42" />

<img width="500" height="238" alt="image" src="https://github.com/user-attachments/assets/1f60045f-fab2-40b2-bc32-b3316c4de21e" />

<img width="521" height="275" alt="image" src="https://github.com/user-attachments/assets/05e1b6c3-bb9f-4186-b856-0beff7a38dad" />

Example:

 Home Page
 Dataset Upload
 VPN Detection Results
 Attack Detection Dashboard
 Threat Analytics
 Concept Drift Monitoring



Future Improvements

 Deep Learning Models (CNN, LSTM, Transformer)
 Real-time Network Packet Capture
 Cloud Deployment (AWS/Azure)
 REST API Integration
 Automatic Model Retraining


 Author

Pragati Tiwari

Bachelor of Technology (Computer Science & Engineering)



