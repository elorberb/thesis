# Use Python 3.9 image from Docker Hub as the base image
FROM python:3.9

# Expose port 8080 for the Streamlit app
EXPOSE 8080

# Set the working directory in the container to /app
WORKDIR /app

# Copy the Python requirements file to the container
COPY requirements.txt ./requirements.txt

# Install the Python dependencies
RUN pip3 install -r requirements.txt

# Copy the rest of the application's code to the container
COPY . .

# Command to run the Streamlit app
CMD ["streamlit", "run", "☘️_Introduction.py", "--server.port", "8080"]
