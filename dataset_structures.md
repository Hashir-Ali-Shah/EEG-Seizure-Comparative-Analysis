# Understanding the EEG Dataset Structures

This document provides a simple, jargon-free reference guide explaining how each of the three EEG datasets is structured and how they are formatted so a Logistic Regression model can read them.

---

## Dataset 1: BEED Bangalore EEG Epilepsy Dataset

### 1. What is the Structure?
* **Type**: Pure Tabular Data (saved as a standard `.csv` file).
* **Format**: A flat 2D grid containing **8,000 rows** and **16 columns**.

### 2. What do the Rows and Columns Mean?
* **Each Row (Sample)**: Represents a single EEG recording sample (one observation of a patient's brain activity during a specific window).
* **Each Column (Feature)**: Represents one of 16 different pre-calculated statistical numbers (like the average signal level, how much the signal fluctuates, etc.). 
* **Key Insight**: There is **no raw brainwave signal** here. The raw signals were already converted into stats before this dataset was saved.

### 3. How do we feed it to Logistic Regression?
* **No feature extraction is needed.**
* Since it is already a flat table (rows and 16 columns), we just load the 16 columns of numbers and feed them directly into the model.

---

## Dataset 2: Epileptic Seizure Recognition Dataset

### 1. What is the Structure?
* **Type**: Flattened Single-Channel Time Series (saved as a `.csv` file).
* **Format**: A 2D grid containing **11,500 rows** and **178 columns**.

### 2. What do the Rows and Columns Mean?
* **Each Row (Sample)**: Represents one complete, continuous EEG recording session for a patient.
* **Each Column (Time Point)**: Represents the value of the brainwave signal at a specific, sequential point in time (from time point 1 to time point 178).
* **Key Insight**: Each row is a single continuous wave recorded by one sensor over 178 consecutive moments in time.

### 3. How do we feed it to Logistic Regression?
* **No feature extraction is needed.**
* Even though this is time-series data, it is already laid out flat for us in the CSV. We simply treat the 178 time points as 178 individual features and feed them directly to the model.

---

## Dataset 3: EEG Seizure Analysis Dataset

### 1. What is the Structure?
* **Type**: 3D Multi-Channel Time Series (saved as a binary `.npz` file).
* **Format**: A raw 3D recording block of **8,282 patient samples**, **23 channels**, and **250 time points** per channel.

### 2. What do the Rows, Channels, and Time Points Mean?
* **Each Row (Sample)**: Represents one patient's recording session.
* **Each Channel (Sensor)**: Represents a **different physical sensor (electrode)** placed on a different part of the patient's head (like the forehead, the back of the head, etc.). 
  * *Each channel records different brain activity from its specific location at the exact same time.*
* **Each Time Point**: A single value recorded by a specific sensor at a specific moment in time (over a window of 250 moments).
* **Raw Size**: Multiplying this out gives $23 \text{ channels} \times 250 \text{ time points} = 5,750 \text{ raw numbers}$ per patient.

### 3. How do we feed it to Logistic Regression?
* **Feature Extraction is Required.**
* Logistic Regression cannot accept a 3D signal block directly. It requires a flat list of columns. 
* To resolve this, we perform feature extraction:
  1. We take each of the 23 physical sensors.
  2. For each sensor's 250 recorded time points, we calculate **6 simple statistics**: *Mean, Standard Deviation, Maximum, Minimum, Skewness, and Kurtosis*.
  3. This leaves us with exactly 6 clean numbers per sensor instead of 250.
  4. We stack them all together: $23 \text{ sensors} \times 6 \text{ statistics} = 138 \text{ total features}$.
* This turns the complex 3D recording into a flat list of 138 features that Logistic Regression can easily read and analyze.
