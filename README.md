# Dokkyo NeuroNavigation System

## 1. Objective

The aim of this repository is to provide the software and analysis tools for the AprilTag-based neuro-navigation system developed at Dokkyo Medical University and Tokyo Metropolitan College of Industrial Technology.

The system is designed to support neuro-navigation using AprilTag-based tracking, coordinate registration, individual head-surface geometry, and standardized brain-space transformation.

Certain technologies implemented in this system are the subject of pending patent applications.

Please refer to `LICENSE.txt` for conditions of use, including restrictions related to commercial use, redistribution, and patent rights.

## 2. Features

The system includes functions for:

* AprilTag-based position and orientation tracking using an Android application
* Registration between physical and anatomical coordinate systems
* Integration of individual head-surface data acquired using a 3D scanner
* Transformation of coordinates into standardized MNI brain space
* Estimation of stimulation locations on scalp and brain surfaces
* Calculation and visualization of motor evoked potential (MEP) maps
* Export of MEP mapping results as NIfTI (`.nii.gz`) files
* OSC-based communication of coil and target information to external applications

Some components, including the Android and Unity-based applications, are distributed only in compiled form.

The exact functions available may differ depending on the version of this repository.

---

## 3. Requirements

The required software and hardware depend on the configuration of the system.

### 3.1 AprilTag Tracking System

The following components are required for AprilTag-based tracking:

* Android smartphone

  * Android is currently the only supported mobile platform.

* Printed AprilTags

  * Tag family: `tag41h12`
  * For the recommended tag IDs, sizes, layout, and printing specifications, please refer to `tag.pdf`.

* Wi-Fi network

  * The Android smartphone and the computer running the MEP recording system must be connected to the same network.
  * UDP communication between the devices must be available.
  * Please also refer to the provided LabVIEW VI file for the MEP recording configuration.

### 3.2 Registration and Analysis System

A computer capable of running Python and the required libraries is necessary.

The system has been tested in the following environment:

* Operating system: macOS 26.4.1 (25E253)
* Computer: Mac Studio (2022)
* Processor: Apple M1 Max
* Python: 3.11.13

The software may also run on other operating systems and hardware configurations; however, these environments have not been fully validated.

### 3.3 Python Environment

Python is required to run the registration and MEP mapping analysis software.

Tested environment:

```text
Python 3.11.13
```

Required Python packages are listed in `requirements.txt`.

The tested package versions are:

```text
numpy==2.4.2
pandas==2.3.3
scipy==1.16.3
scikit-learn==1.8.0
matplotlib==3.9.4
nibabel==5.4.0
pyvista==0.47.1
networkx==3.6.1
chardet==5.2.0
```

Install the required packages using:

```bash
pip install -r requirements.txt
```

### 3.4 Additional Hardware and Software

Depending on the intended application, the following components may also be required:

* 3D scanner for acquisition of individual head-surface geometry
* LabVIEW and the provided VI file for MEP recording
* AprilTag markers for tracking the head and stimulation device
* Computer capable of receiving UDP / OSC data
* Transcranial magnetic stimulation (TMS) system when the software is used for TMS neuronavigation

---

## 4. How to Use

### 4.1 Prepare the AprilTags

Print the required AprilTags according to the specifications described in `tag.pdf`.

Make sure that:

* the correct `tag41h12` IDs are used;
* the printed dimensions are accurate; and
* the tags are mounted on a flat and rigid surface.

The physical size of each tag is important because AprilTag-based pose estimation depends on the specified tag dimensions.

### 4.2 Install the Android Application

Install the provided APK file on an Android smartphone.

The Android application detects the AprilTags and obtains their position and orientation information.

The Android application is distributed as a compiled APK file. Source code for this component is not included in this repository unless otherwise stated.

[https://youtu.be/_zS6yGBMMZs]

### 4.3 Configure the Network

Connect the Android smartphone and the computer running the MEP recording system to the same Wi-Fi network.

Configure the destination IP address and UDP port according to the network settings of the receiving computer.

Make sure that:

* both devices are on the same network;
* UDP communication is allowed;
* the selected port is not blocked by the operating system firewall; and
* the destination IP address is correctly configured.

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

Acquire the participant's head-surface geometry using a compatible 3D scanner.

The following anatomical landmarks should be identified on the head model:

* Nasion (`Nz`)
* Inion (`Iz`)
* Left auricular point (`A1`)
* Right auricular point (`A2`)
* Vertex (`Cz`)

These landmark coordinates should be marked and saved as a *.pp file using MeshLab (https://www.meshlab.net/).

<img width="2558" height="1342" alt="image" src="https://github.com/user-attachments/assets/6f1f16cb-3120-4463-8304-91bc59287f4d" />


Additional stimulation or control points may also be defined depending on the intended registration method.

The head-surface mesh and corresponding landmark information are used to establish the transformation between the physical tracking coordinate system and anatomical space.

### 4.6 Registration

The registration software aligns the smartphone-based AprilTag coordinate system with the coordinate system of the individual 3D head model.

Depending on the analysis configuration, registration may be performed using:

* Anatomical landmarks
* Stimulation control points
* A combination of anatomical landmarks and control points
* Standard MNI152 anatomy without participant-specific 3D scanner data

The combined landmark and control-point approach corresponds to the hybrid registration method used in the associated study.

### 4.7 MEP Recording

MEP signals can be recorded using the provided LabVIEW VI.

The AprilTag tracking information can be recorded together with the MEP data so that each stimulation trial can subsequently be associated with the corresponding coil and target coordinates.

Please refer to the provided VI file for detailed acquisition settings.

### 4.8 Run the MEP Mapping Analysis

After MEP recording and registration, run the Python-based MEP mapping analysis.

The analysis integrates:

* MEP amplitudes
* Coil coordinates
* Target coordinates
* Anatomical landmarks
* Individual head-surface geometry
* Coordinate transformation parameters

The analysis can be executed directly from the command line:

```bash
python MEP_mapping.py
```

Alternatively, the `MEP_mapping` class can be imported and executed from another Python script:

```python
import MEP_mapping as mep

MEPs = mep.MEP_mapping(
    MEP_path,
    savePath,
    subj_num=int(subj_num),
    isFullmode=am,
    obj_data=obj_data,
    modes=mode
)
```

The `modes` argument specifies the registration method. Available options are:

* `"landmark"` — registration using anatomical landmarks
* `"ControlPoints"` — registration using stimulation control points
* `"Both"` — registration using both anatomical landmarks and control points
* `"NoScanner"` — analysis using the standard MNI152 anatomical model without participant-specific 3D scanner data

The `isFullmode` argument specifies the transformation model:

* `True` — full affine transformation
* `False` — restricted transformation without full affine deformation

The stimulation locations are transformed into anatomical and MNI coordinate systems.

MEP amplitudes are spatially interpolated across the relevant scalp and brain surfaces.

The analysis can export results including:

* stimulation coordinates;
* estimated target coordinates;
* registration errors;
* coil-to-target errors;
* MEP amplitudes;
* weighted MEP centroid coordinates; and
* MEP maps in NIfTI (`.nii.gz`) format.

---

## 5. Communication

### 5.1 Overview

The navigation application supports OSC (Open Sound Control) communication over UDP.

The compiled Unity application acts as an OSC sender and can transmit coil, target, and tracking information to external software.

Default configuration:

* Protocol: OSC over UDP
* Default port: `8080`
* Default destination IP address: `127.0.0.1`
* Destination IP address: configurable in the application

The operating system firewall and network configuration must allow UDP communication on the selected port.

### 5.2 OSC Messages

The following OSC messages are transmitted:

| OSC Address       | Data                  | Description                                                 |
| ----------------- | --------------------- | ----------------------------------------------------------- |
| `/OSC/diff`       | `float, float, float` | Difference between coil and target positions in millimeters |
| `/OSC/target_pos` | `float, float, float` | Target position                                             |
| `/OSC/coil_pos`   | `float, float, float` | Coil position                                               |
| `/OSC/coil_rot`   | `float, float, float` | Coil orientation                                            |
| `/OSC/Navi`       | `string`              | JSON-formatted navigation and tracker information           |

### 5.3 Coil-Target Difference

The values transmitted through `/OSC/diff` are calculated as:

```text
diff_x = (coil_y - target_y) × 1000
diff_y = (-coil_z - (-target_z)) × 1000
diff_z = (coil_x - target_x) × 1000
```

The transmitted values are expressed in millimeters.

Note that the axis order differs from the original Unity coordinate order.

### 5.4 Navigation Data

`/OSC/Navi` contains a JSON-formatted array of navigation data.

Each entry has the following basic format:

```json
{
    "ID": 1,
    "x": 0.0,
    "y": 0.0,
    "z": 0.0
}
```

Virtual tracker coordinates are expressed relative to the reference tracker.

The following IDs are reserved by the current implementation:

|        ID | Description                   |
| --------: | ----------------------------- |
|   `< 100` | Virtual trackers              |
|     `500` | Coil position                 |
|     `501` | Coil rotation                 |
|     `502` | Target position               |
| `900–905` | Coil surface reference points |

Example:

```json
[
    {
        "ID": 1,
        "x": 0.012,
        "y": -0.034,
        "z": 0.156
    },
    {
        "ID": 500,
        "x": 0.021,
        "y": 0.115,
        "z": -0.043
    },
    {
        "ID": 501,
        "x": 10.2,
        "y": 25.4,
        "z": 2.1
    },
    {
        "ID": 502,
        "x": 0.018,
        "y": 0.110,
        "z": -0.040
    }
]
```

---

## 6. Output Files

Depending on the selected analysis mode, the software may generate the following files:

### NIfTI MEP Map

```text
S{subject}_{mode}_{transform}.nii.gz
```

The file contains the spatially interpolated MEP amplitude distribution in MNI space.

### Coordinate File

```text
S{subject}_{mode}_{transform}.node
```

This file contains the estimated stimulation coordinates and the weighted centroid of the MEP distribution.

### Analysis Information

```text
S{subject}_{mode}_{transform}.json
```

The JSON output may include:

* coil coordinates on the scalp;
* coil coordinates in MNI space;
* stimulation target coordinates;
* MEP amplitudes;
* weighted centroid coordinates;
* registration errors; and
* coil-to-target errors.

---

## 7. Important Notes

Appropriate calibration and registration are required before using this system.

Navigation accuracy may be affected by:

* Camera placement
* AprilTag detection accuracy
* AprilTag printing accuracy
* Physical tag dimensions
* Marker placement
* Network communication
* Calibration accuracy
* Anatomical landmark identification
* Quality of the individual head-surface model
* 3D scanner accuracy
* Coordinate transformation procedures

Users should independently validate the accuracy and reliability of the system for their intended research application.

This software is intended for research purposes and is not provided as a certified medical device.

The navigation results should therefore not be used as the sole basis for clinical diagnosis, treatment decisions, or clinical procedures.

---

## 8. Distribution

This repository may contain both source-code-based analysis tools and compiled software components.

In particular, some software components, including the Android and Unity-based navigation applications, are distributed only in compiled form.

The availability of a compiled application does not imply that its source code, internal implementation, or associated patent rights are made available.

Please refer to `LICENSE.txt` for restrictions regarding:

* commercial use;
* redistribution;
* reverse engineering;
* decompilation;
* modification; and
* patent rights.

---

## 9. Citation

If you use this software or analysis pipeline in academic research, please cite the relevant publication describing the Dokkyo NeuroNavigation System.

Citation information will be added here upon publication.

---

## 10. License

This software is provided primarily for non-commercial academic, research, and educational purposes.

Certain technologies, methods, algorithms, and system configurations implemented in this software are the subject of pending patent applications.

Permission to access or use the distributed software does not automatically grant any rights under patents or pending patent applications.

Some components are distributed only as compiled software.

Please read `LICENSE.txt` before using, copying, modifying, redistributing, reverse engineering, or otherwise utilizing any component of this system.

---

## 11. Contact

For questions regarding:

* research use;
* collaborative research;
* commercial licensing;
* patent-related permissions; or
* technical issues,

please contact the authors.
