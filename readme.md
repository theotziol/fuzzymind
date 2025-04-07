# FCM-APP

FCM-APP offers a flexible and intuitive user interface for Fuzzy Cognitive Map (FCM) construction, analysis, visualization, and weight matrix optimization. The software is entirely written in Python, utilizing open-source packages such as Streamlit, TensorFlow, Pandas, Numpy, NetworkX, and Matplotlib.
In the current version, the software offers the following features:

1. Manual construction of both numeric and linguistic FCMs
2. Automatic construction of FCMs
3. Knowledge aggregation of linguistic FCMs
4. FCM graph visualization
5. Data preprocessing tools
6. FCM learning for classification tasks (Neural-FCM classifier)
7. FCM learning for regression tasks (Neural-FCM regressor)
Demos for these features can be found in the Manual.docx file.  
  
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

#### b. Install required Python libraries using the `requirements.txt` file

```sh
pip install --use-deprecated=legacy-resolver -r App\requirements.txt
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

### 2. Create a Dockerfile

Add the following `Dockerfile` in your project directory:

```dockerfile
# Use an official Python runtime as a parent image
FROM python:3.10

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the Streamlit port
EXPOSE 8501

# Command to run the Streamlit app
CMD ["streamlit", "run", "Home.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### 3. Build and Run the Docker Container

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

This project is licensed under the MIT License. See the `LICENSE` file for details.

