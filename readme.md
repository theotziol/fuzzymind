# FuzzyMind

**FuzzyMind** app offers a flexible and intuitive user interface for Fuzzy Cognitive Map (FCM) construction, analysis, visualization, and weight matrix optimization. The software is entirely written in Python, utilizing open-source packages such as Streamlit, TensorFlow, Pandas, Numpy, NetworkX, and Matplotlib.
In the current version, the software offers the following features:

1. Manual construction of both numeric and linguistic FCMs
2. Automatic construction of FCMs
3. Knowledge aggregation of linguistic FCMs
4. FCM graph visualization
5. Data preprocessing tools
6. FCM learning for classification tasks (Neural-FCM classifier)
7. FCM learning for regression tasks (Neural-FCM regressor)

Demos for these features can be found in the `Manual.docx` file. A publication manuscript is under internal revision.

**Developed by [ACTA Lab Team](https://acta.energy.uth.gr/)**

## Installation Instructions (Local Setup)

To install and run the Streamlit application locally, follow these steps:

### 1. Set Up a Virtual Environment

It is recommended to install the required Python libraries in a virtual environment to avoid conflicts with other installed libraries.

#### a. Open a command prompt and navigate to your desired location

```sh
cd Desktop
```

#### b. Create a virtual environment named `App`

```sh
python -m venv App
```

#### c. Activate the virtual environment

```sh
App\Scripts\activate
```

### 2. Install Dependencies

#### a. Unzip the `fcm-app-master.zip` file inside the `App` folder

#### b. Install required Python libraries using the `requirements-gpu.txt` file

```sh
pip install --use-deprecated=legacy-resolver -r App\requirements-gpu.txt
```

### 3. Run the Application

#### a. Change directory to the `App` folder

```sh
cd App
```

#### b. Start the Streamlit app:

```sh
streamlit run Home.py
```

The application will launch in your default web browser at:

```
http://localhost:8501/
```

---

## Running the App with Docker

To run this application inside a Docker container, follow these steps:

### 1. Install Docker

Make sure you have Docker installed on your system. You can download and install Docker from [here](https://www.docker.com/get-started).


### 2. Build and Run the Docker Container

#### a. Build the Docker image

```sh
docker build -t fcm-app .
```

#### b. Run the container

```sh
docker run -p 8501:8501 fcm-app
```

Now, access the application in your web browser at:

```
http://localhost:8501/
```

---

## Contributing

Feel free to open an issue or submit a pull request if you have any suggestions or improvements!

## License

This project is licensed under APACHE 2.0. See the `LICENSE.txt` file for details. 

## Caveats

The app was developed with TensorFlow version 2.9.1 which supports CUDA acceleration on Windows, and Streamlit version 1.33.0. These 2 libraries have an unmatched dependency, the protobuf library. Using **--use-deprecated=legacy-resolver** in **pip install** command currently solves this issue. *If you are running a different OS, or you don't have/need CUDA-GPU deep learning accelerations, consider installing the latest versions with the `requirements.txt` file*.  

## Found us

This app was developed by Theodoros Tziolas during his PhD studies and under the guidance of Professor [Elpiniki Papageorgiou](https://www.energy.uth.gr/www/index.php/en/personnel/papageorgiou-elpiniki) and is a property of the [ACTA Lab](https://acta.energy.uth.gr/). Visit our lab website to discover more great projects or to request collaboration. 