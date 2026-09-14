# Dokkyo NeuroNavigation System

## 1. Objective

The aim of this repository is to publish the source code of the AprilTag-based neuro-navigation system developed at Dokkyo Medical University.

This system is designed to support neuro-navigation using AprilTag-based tracking and coordinate registration techniques.

Certain technologies implemented in this system are the subject of pending patent applications.

Please refer to `LICENSE.txt` for conditions of use, including restrictions related to commercial use and patent rights.

## 2. Features

The system includes functions for:

* AprilTag-based position and orientation tracking
* Registration between physical and anatomical coordinate systems
* Integration of individual head surface data
* Transformation of coordinates into standardized brain space
* Estimation and visualization of neuro-navigation targets

The exact functions available may differ depending on the version of the repository.

## 3. Requirements

The required software and hardware depend on the configuration of the system.

Typical requirements include:

* Python
* AprilTag detection library
* Compatible camera device
* Required Python packages
* Individual head surface or anatomical data
* Appropriate tracking markers and experimental hardware

Install the required Python packages using:

```bash
pip install -r requirements.txt
```

## 4. How to Use

### 4.1 Clone the Repository

```bash
git clone <repository-url>
cd <repository-name>
```

### 4.2 Install Dependencies

```bash
pip install -r requirements.txt
```

### 4.3 Prepare the Experimental Environment

Before running the system, prepare the required hardware and data.

Depending on the experimental setup, this may include:

* Placement of AprilTags
* Camera configuration
* Calibration of the tracking environment
* Acquisition or loading of individual head surface data
* Registration of anatomical landmarks
* Definition of target coordinates

### 4.4 Configure the System

Set the parameters required for your experimental environment.

These may include:

* Camera parameters
* AprilTag IDs and marker sizes
* Network settings
* Coordinate transformation parameters
* Anatomical landmark coordinates
* Target coordinates

Refer to the configuration files and comments in the source code for details.

### 4.5 Run the System

Run the appropriate main program for your configuration.

For example:

```bash
python main.py
```

The exact execution command may differ depending on the version and experimental configuration.

## 5. Important Notes

Appropriate calibration and registration are required before using this system.

Navigation accuracy may be affected by:

* Camera placement
* AprilTag detection accuracy
* Marker placement
* Calibration accuracy
* Anatomical registration accuracy
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

For questions regarding:

* Research use
* Collaborative research
* Commercial licensing
* Patent-related permissions
* Technical issues

please contact the authors or the relevant office at Dokkyo Medical University.
