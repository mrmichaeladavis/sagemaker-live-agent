<!-- BEGIN_TF_DOCS -->
# SageMaker Hugging Face real-time endpoint deployment

<!-- markdownlint-disable MD024 -->
This sample demonstrates how to deploy and interact with a model supported by the Hugging Face LLM Inference Container for Amazon SageMaker.

Specifically, this sample deploys a SageMaker real-time endpoint, hosting [Mistral-7B-Instruct-v0.1](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1) from Hugging Face.

> **Warning**
> This sample allows you to interact with models from third party providers. Your use of the third-party generative AI (GAI) models is governed by the terms provided to you by the third-party GAI model providers when you acquired your license to use them (for example, their terms of service, license agreement, acceptable use policy, and privacy policy).
>
> You are responsible for ensuring that your use of the third-party GAI models comply with the terms governing them, and any laws, rules, regulations, policies, or standards that apply to you.
>
> You are also responsible for making your own independent assessment of the third-party GAI models that you use, including their outputs and how third-party GAI model providers use any data that might be transmitted to them based on your deployment configuration. AWS does not make any representations, warranties, or guarantees regarding the third-party GAI models, which are "Third-Party Content" under your agreement with AWS. This sample is offered to you as "AWS Content" under your agreement with AWS.

## Prerequisites

An AWS account. We recommend you deploy this solution in a new account.

- AWS CLI: configure your credentials

```
aws configure --profile [your-profile]
AWS Access Key ID [None]: xxxxxx
AWS Secret Access Key [None]:yyyyyyyyyy
Default region name [None]: us-east-1
Default output format [None]: json
```

- Terraform: v1.9.8 or greater
- Make sure you have sufficient quota for the instance type implemented in this sample (service Amazon SageMaker, instance type ml.g5.2xlarge for endpoint usage). For more information, refer to [AWS service quotas](https://docs.aws.amazon.com/general/latest/gr/aws_service_limits.html).
- [A Hugging Face account](https://huggingface.co/welcome)
- A Hugging Face API token. Mistral models are now gated on Hugging Face. To get access, you need to create a user access token. The procedure is detailed at: [https://huggingface.co/docs/hub/security-tokens](https://huggingface.co/docs/hub/security-tokens)
- Accept to share you contact information: The model deployed in this sample requires you to agree to share your information before you can access it. Once logged in, visit the [model page](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1) and click on the button 'Agree and access repository'.

## Deploy

This project is built using [Terraform](https://www.terraform.io/). See [Getting Started - AWS](https://developer.hashicorp.com/terraform/tutorials/aws-get-started) for additional details and prerequisites.

1. Clone this repository.

```shell
git clone https://github.com/mrmichaeladavis/sagemaker-live-agent.git
cd sagemaker-live-agent
```

1. Enter the terraform directory.

```shell
cd terraform
```

1. Initialize the neccessary Terraform providers

```shell
terraform init
```

1. Update your API Access token

Navigate to the [main file](./main.tf) and update the value of the variable ***HF\_API\_TOKEN*** to use the value of the user access token you created in the pre-requisites.

1. Check the plan.

```shell
terraform plan
```

1. Deploy the sample in your account.

```shell
terraform apply
```

With the default configuration of this sample, the observed deployment time was ~7minutes 30s.

To protect you against unintended changes that affect your security posture, the Terraform prompts you to approve before deploying them. You will need to answer ***yes*** to get the solution deployed.

![Hugging Face](./docs/deployed\_endpoint.png)

## Test

You can then invoke the provisioned endpoint. For instance, we show here an example using [boto3](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/sagemaker-runtime/client/invoke_endpoint.html) in Python.

```python
import boto3
import json
import os

runtime= boto3.client('runtime.sagemaker')

ENDPOINT_NAME = "mistralendpoint"

dic = {
    "inputs": "<s>[INST] write the recipe for a mayonnaise [/INST]",
    "parameters": {
        "temperature": 0.6,
        "top_p": 0.95,
        "repetition_penalty": 1.2,
        "top_k": 50,
        "max_new_tokens": 4000,
        "stop": ["</s>"]
    }
}

response = runtime.invoke_endpoint(EndpointName=ENDPOINT_NAME,
                                ContentType='application/json',
                                Body=json.dumps(dic))

result = json.loads(response['Body'].read().decode())
print(result)
```

Example of output:

```json
[{'generated_text': '<s>[INST] write the recipe for a mayonnaise [/INST] Here is a simple recipe for making your own mayonnaise:\nIngredients:\n\n* 1 egg yolk\n* 2 tablespoons of white vinegar or lemon juice\n* 1 teaspoon mustard (optional)\n* 1 cup vegetable oil, such as canola or light olive oil\n* Salt and pepper to taste\n\nInstructions:\n\n1. In a small bowl, whisk together the egg yolk, vinegar, and mustard until well combined.\n2. Slowly drizzle in the oil while continuing to whisk until it forms a smooth emulsion. This should take about 5-7 minutes.\n3. Taste the mixture and adjust seasoning with salt and pepper if needed.\n4. Cover the bowl with plastic wrap and refrigerate for at least an hour before using. The longer you let it sit, the better it will be. It will thicken up a bit and become more flavorful.'}]
```

## Clean up

```shell
terraform destroy
```

Delete all the associated logs created by the different services in Amazon CloudWatch logs.

## Requirements

| Name | Version |
|------|---------|
| <a name="requirement_terraform"></a> [terraform](#requirement\_terraform) | >= 1.0.7 |
| <a name="requirement_aws"></a> [aws](#requirement\_aws) | ~>5.0 |
| <a name="requirement_random"></a> [random](#requirement\_random) | >= 3.6.0 |

## Providers

No providers.

## Modules

| Name | Source | Version |
|------|--------|---------|
| <a name="module_sagemaker-endpoint"></a> [sagemaker-endpoint](#module\_sagemaker-endpoint) | ../.. | n/a |

## Resources

No resources.

## Inputs

| Name | Description | Type | Default | Required |
|------|-------------|------|---------|:--------:|
| <a name="input_region"></a> [region](#input\_region) | AWS region to deploy the resources | `string` | `"us-east-1"` | no |

## Outputs

No outputs.
<!-- END_TF_DOCS -->