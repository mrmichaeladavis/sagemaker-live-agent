# SageMaker Live Agent

## Situation

Model providers like Eleven Labs and other companies don't want to share their models (intellectual property) with us and we prefer to not share our data with them removing the option of accessing their models via a vendor owned and managed SaaS platform.

A solution is to leverage the SageMaker endpoint API which allows for the creation of an isolated VPC which has an interface to it of the SageMaker API to third-party models like Eleven Labs.

The goal of this is to show that we can support running models within a SageMaker endpoint with 3rd party models as well as open source models like Kokoro TTS.

## This repo

The goal of this repo is to stand up a simulated virtual agent which leverages an audio model within a SageMaker endpoint and an interface to be able to talk and chat with that model. It also interfaces with external models like OpenAI to show how the capabilities of this solution would work in a production deployment.

Use over that endpoint API something like the OpenAI API interface.

## Additional docs to read

<https://aws.amazon.com/blogs/machine-learning/create-a-sagemaker-inference-endpoint-with-custom-model-extended-container/> How to crate a custom container for SageMaker endpoint

## Deployment via Terraform

The Terraform sets up the entire environment and attempts to emulate what a production environment would look like once deployed. It leverages a custom-built SageMaker container that is instantiated through the SageMaker endpoint API.

It is based off the terraform module from <https://github.com/aws-ia/terraform-aws-sagemaker-endpoint.git>

## Sagemaker Live Agent

The web based live agent is launched via a sagemaker endpoint and is powered by [LiveKit Agent](https://github.com/livekit/agents) and WebRTC. It is based off the project at <https://github.com/agonza1/webrtc-agent-livekit> with significant modifications to support the SageMaker endpoint and more modular design to support multiple TTS and STT providers.

This project is a quickstart template to run locally or with 3rd party integrations. It showcases how to combine WebRTC, LiveKit’s Agent framework, and open-source tools like Whisper and Llama to prototype low-latency voice assistants for real-time applications.

## 🧠 What’s Inside

- 🌐 **WebRTC + LiveKit**: Real-time media transport with WebRTC powered by LiveKit.
- 🤖 **LiveKit Agent**: Modular plugin-based framework for voice AI agents.
- 🗣️ **STT + TTS Support**: Plug in Whisper, Deepgram, ElevenLabs, or others via SageMaker Endpoint or direct using their APIs.
- 💬 **LLM Integration**: Use open source models via SageMaker Endpoint API or connect to AWS/ OpenAI / Anthropic APIs.
- 🧪 **Local Dev**: Run everything locally with Docker Compose or Python virtual env.

[fast-preresponse.py](./agent-worker/fast-preresponse.py) using 3rd party services and the complete metrics capture in place.

More info in [Agent Worker README](./agent-worker/README.md).

## 🚀 Quick Start (Local) - NO AWS

YOU MUST RUN THE AGENTS PLAYGROUND ON LOCALHOST it will not work if you run it remotely as it won't get rights to access microphone.

Change the node IP in livekit.yaml if you are not running on localhost.

1. Clone:

```bash
# Clone the repo
git clone https://github.com/mrmichaeladavis/sagemaker-live-agent.git
cd sagemaker-live-agent
```

1. Install dependencies docker and docker compose

1. If you want to also run the example frontend, copy and rename the [`.env.example`](./agents-playground/.env.example) file to `.env.local` and fill in the necessary environment variables. You can also update the YML files to configure the different services. For example, agents-playground.yml:

```bash
LIVEKIT_API_KEY=<your API KEY> # change it in livekit.yaml
LIVEKIT_API_SECRET=<Your API Secret> #change it in livekit.yaml
NEXT_PUBLIC_LIVEKIT_URL=ws://localhost:7880 #wss://<Your Cloud URL>
```

1. Run docker-compose:

```bash
  docker compose up --build
```

Make sure that at least the services "agent-playground", "agent-worker", "livekit" and "redis" in the docker-compose are uncommented and the envs are updated.

1. Open [http://localhost:3000](http://localhost:3000) with your browser to see the result.

1. Connect to a room

## Monitoring

The solution is build using Prometheus and grafana. The end to end flow is:
Agent worker → writes metrics to shared temp folder → agent_metrics service exposes them → Prometheus scrapes them → Grafana displays them.

[Agent Worker](./agent-worker/) live metrics are exposed on port 9100 and can be accessed at:

```bash
http://localhost:9100/metrics
```

Grafana is available in [http://localhost:3001](http://localhost:3001) with default user/password: admin/admin
A default dashboard is setup to visualize basic real time voice agents information like audio and llm latency.

## 🙏 Credits

This project is built on top of amazing open-source tools and services:

- **[LiveKit](https://livekit.io/) and [LiveKit Agents](https://github.com/livekit/agents)** - WebRTC Framework for building voice AI agents
- **[Prometheus](https://prometheus.io/) and [Grafana](https://grafana.com/)** - Metrics collection, monitoring and visualization
