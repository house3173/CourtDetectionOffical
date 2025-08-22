# Nhận diện cấu trúc sân trong môn tennis và cầu lông

## Cấu trúc thư mục
```text
CourtDetectionOffical/
│
├── images/
│
├── outputs/
│
├── evaluate/
│   ├── data_examples/
│   │   ├── images/
│   │   ├── ground_truth.json
│   │   └── predicted.json
│   ├── oks_metrics.py
│   ├── pck_metrics.py
│   ├── rmse.py
│   └── evaluate.py
│
├── models/
│
├── src/
│   ├── court_resnet50.py
│   ├── court_detection.py
│   └── court_detection_video.py
│
├── Court Detecttion Docs.docx
├── requirements.txt
└── README.md
