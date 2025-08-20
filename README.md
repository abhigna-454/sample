# Network Anomaly Detection using Machine Learning

This project detects anomalies in network traffic using the NSL-KDD dataset. 
It classifies traffic into five categories:
- Normal
- DoS (Denial of Service)
- Probe (Surveillance/Probing)
- R2L (Remote to Local)
- U2R (User to Root)

## Files
- `network_anomaly_detection.py`: Main script
- Dataset is loaded automatically from GitHub (NSL-KDD).

## How to Run (Google Colab)
1. Upload these files to your GitHub repo.
2. Open Google Colab and clone your repo:
   ```bash
   !git clone https://github.com/yourusername/yourrepo.git
   ```
3. Navigate into the project folder and run:
   ```bash
   %cd yourrepo
   !python network_anomaly_detection.py
   ```

## Output
- Accuracy of the model
- Classification report (Precision, Recall, F1-score)
- Confusion matrix (plotted as heatmap)
- Sample predictions showing actual vs predicted classes
