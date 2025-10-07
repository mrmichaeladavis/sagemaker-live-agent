import json

import boto3

# Name of your SageMaker endpoint (replace with actual endpoint name)
endpoint_name = "mistralendpoint"

# Create SageMaker runtime client
sm_runtime = boto3.client("sagemaker-runtime")

# Example prompt to send to the Mistral7b model
prompt = "Explain the concept of zero-knowledge proofs in cryptography."

# Construct the payload as expected by the endpoint (JSON format)
payload = {"inputs": prompt, "parameters": {"max_new_tokens": 128, "do_sample": True}}

# Make the inference request
response = sm_runtime.invoke_endpoint(
    EndpointName=endpoint_name, ContentType="application/json", Body=json.dumps(payload)
)

# Parse the response from the model
result = response["Body"].read().decode("utf-8")
print("Model response:", result)
