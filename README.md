# AAP_Essentials

AAP_Essentials is a set of applications designed to detect PII (Personally Identifiable Information) leaks and identify inappropriate or absurd content within text using advanced language models. It integrates multiple components, including a FastAPI backend for hosting the language models and a deep learning-based model to ensure content integrity.

## Project Structure

The project is divided into two primary folders, each serving a specific function:

### `bot/` - FastAPI Application for PII Detection and Content Moderation
This folder contains the FastAPI application responsible for:
- Hosting a Language Model (LLM) capable of identifying and classifying text that contains potential PII leaks or absurd, inappropriate content.
- Returning structured output that highlights detected issues in the text, such as identifying PII or flagging ridiculous or harmful posts.
- Making requests to another LLM for additional analysis or to trigger further actions based on detected content.

### `model/` - FastAPI Application for Deep Learning Neural Network Model
This folder contains a FastAPI application that serves the deep learning neural network model, responsible for:
- Processing textual data and providing predictions based on deep learning techniques.
- Acting as the core of content analysis, ensuring that the model's predictions are accurate and reliable.

## Features

- **PII Leak Detection**: The bot identifies and flags potential personal information leaks, helping to ensure compliance with privacy standards.
- **Content Moderation**: Flags posts or text that are absurd, ridiculous, or inappropriate for various use cases (e.g., social media platforms).
- **Structured Output**: The output of the LLMs is structured for easy interpretation, including identified issues and suggested actions.
- **Model Integration**: FastAPI integrates multiple models to provide robust analysis of incoming content.
