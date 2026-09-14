# Dokkyo NeuroNavigation System（まだまだ・・・）

## 1. Objective

The aim of this repository is to publish the source code of the AprilTag-based neuro-navigation system developed at Dokkyo Medical University and Tokyo Metropolitan College of Industrial Technology.

This system is designed to support neuro-navigation using AprilTag-based tracking and coordinate registration techniques.

Certain technologies implemented in this system are the subject of pending patent applications.

Please refer to `LICENSE.txt` for conditions of use, including restrictions related to commercial use and patent rights.

## 2. Features

The system includes functions for:

* AprilTag-based position and orientation tracking using an Android application
* Registration between physical and anatomical coordinate systems
* Integration of individual head surface data
* Transformation of coordinates into standardized brain space
* Estimation and visualization of neuro-navigation targets

The exact functions available may differ depending on the version of the repository.

## 3. Requirements

The required software and hardware depend on the configuration of the system.

### 3.1 AprilTag Tracking System

The following components are required for AprilTag-based tracking:

* Android smartphone

  * Android is currently the only supported mobile platform.
* Printed AprilTags

  * Tag family: `tag41h12`
  * For the recommended tag size, layout, and printing specifications, please refer to `tag.pdf`.
* Wi-Fi network

  * The smartphone and the computer running MEP recording system must be connected to the same network (see VI file).
  * UDP communication between the devices must be available.

### 3.2 Registration System

A computer capable of running Python and the required libraries is necessary.

The system has been tested in the following environment:

* Operating system: macOS 26.4.1
* Computer: [Mac model]
* Processor: [Apple Silicon / Intel processor]
* Python: [Python version]

The software may also run on other operating systems and hardware configurations; however, these environments have not been fully validated.

### 3.3 Python Environment

Python is required to run the registration and neuro-navigation software.

Recommended environment:

```text
Python: [version]
```

Required Python packages are listed in `requirements.txt`.

Install the dependencies using:

```bash
pip install -r requirements.txt
```

### 3.4 Additional Hardware

Depending on the intended application, the following hardware may also be required:

* 3D scanner for acquisition of individual head surface geometry
* Computer capable of receiving UDP data from the Android application
* AprilTag markers for tracking the head and stimulation device
* Transcranial magnetic stimulation (TMS) system, when the software is used for TMS neuronavigation

## 4. How to Use

### 4.1 Prepare the AprilTags

Print the required AprilTags according to the specifications described in `tag.pdf`.

Make sure that:

* the correct `tag41h12` IDs are used;
* the printed dimensions are accurate; and
* the tags are mounted on a flat and rigid surface.

### 4.2 Install the Android Application

Install the provided APK file on an Android smartphone.

The Android application detects the AprilTags and sends their position and orientation data to the computer via UDP communication.

### 4.3 Configure the Network

Connect the Android smartphone and the computer to the same Wi-Fi network.

Configure the destination IP address and UDP port in the Android application according to the network settings of the computer running the neuro-navigation software.

Make sure that UDP communication is not blocked by the operating system firewall or network configuration.

### 4.4 Install the Python Environment

Clone this repository:

```bash
git clone https://github.com/shun-irie/TMS_neuronavigation_2026ver.git
cd TMS_neuronavigation_2026ver
```

Install the required Python packages:

```bash
pip install -r requirements.txt
```

### 4.5 Prepare Individual Head Data

Prepare the individual head surface data required for registration.

The head surface geometry can be obtained using a compatible 3D scanner.

Anatomical landmarks used for registration should be identified according to the procedure described in the relevant documentation.

### 4.6 Run the Neuro-navigation System

Start the Android AprilTag tracking application first.

Then run the Python-based registration and neuro-navigation software on the computer.

The tracking data transmitted from the smartphone are received via UDP and integrated with the individual head surface data and coordinate transformation procedures.

Further details regarding calibration, landmark registration, and execution procedures are described in the corresponding documentation and source code.

## 5. Important Notes

Appropriate calibration and registration are required before using this system.

Navigation accuracy may be affected by:

* Camera placement
* AprilTag detection accuracy
* Tag size and printing accuracy
* Marker placement
* Network communication
* Calibration accuracy
* Anatomical landmark registration
* Quality of the individual head surface data
* Coordinate transformation procedures

Users should independently validate the accuracy and reliability of the system for their intended research application.

This software is intended for research purposes and is not provided as a certified medical device.

## 6. Citation

If you use this software in academic research, please cite the relevant publication describing the Dokkyo NeuroNavigation System.

Citation information will be added here upon publication.

## 7. License

This software is provided primarily for non-commercial academic, research, and educational purposes.

Certain technologies implemented in this software are the subject of pending patent applications.

Use of the source code does not automatically grant any rights under patents or pending patent applications.

Please read `LICENSE.txt` before using, modifying, or redistributing this software.

## 8. Contact

For questions regarding research use, collaborative research, commercial licensing, patent-related permissions, or technical issues, please contact the authors.
