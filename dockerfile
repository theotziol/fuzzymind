FROM python:3.10

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install dependencies
# Run this command if you are on Windows and you want to use GPU compatible TensorFlow
RUN pip install --use-deprecated=legacy-resolver -r requirements-gpu.txt
# Run this command if no GPU is available. It installs all the latest compatible versions
# RUN pip install requirements.txt


# Expose the Streamlit port
EXPOSE 8501

# Command to run the Streamlit app
CMD ["streamlit", "run", "Home.py", "--server.port=8501", "--server.address=0.0.0.0"]