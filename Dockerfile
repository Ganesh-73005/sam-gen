
FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel

# Set the working directory
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the handler file
COPY handler.py .

# Command to run the handler
CMD ["python", "-u", "handler.py"]
