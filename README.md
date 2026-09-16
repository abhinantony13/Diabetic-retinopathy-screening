 # RETINEX – AI-Based Diabetic Retinopathy Pre-Screening

RETINEX is an AI-based explainable diabetic retinopathy pre-screening system designed to support early identification of possible diabetic eye complications.


https://diabeticretinopathy-screening.vercel.app (for live demo......)


The system analyzes retinal fundus images and provides an AI-assisted screening result along with explainability information to help users understand which areas of the retinal image influenced the result.

> **Note:** RETINEX is a pre-screening/support tool and is not a replacement for examination or diagnosis by a qualified eye-care professional.

## Problem Statement

Diabetic Retinopathy (DR) can cause vision loss if it is not detected and managed early. In many rural and underserved areas, access to specialized eye-care services can be limited.

RETINEX aims to provide an accessible preliminary screening workflow using retinal fundus images and explainable AI.

## Proposed Solution

RETINEX provides a simple workflow:

1. Upload a retinal fundus image.
2. Check the quality of the image.
3. Perform AI-assisted diabetic retinopathy screening.
4. Display the screening result.
5. Provide explainability using a Grad-CAM attention map.
6. Present the result in a simple and understandable interface.

## Key Features

- 🩺 Retinal fundus image upload
- 🔍 Automated image-quality assessment
- 🤖 AI-assisted diabetic retinopathy screening
- 🧠 Explainable AI using Grad-CAM
- 📊 Screening result visualization
- 🌐 Web-based interface
- 📱 Designed with accessibility and rural healthcare support in mind

## Explainable AI

RETINEX uses explainability techniques to provide a visual indication of the regions of the retinal image that contributed to the AI model's output.

The Grad-CAM attention map helps make the model's prediction easier to interpret instead of presenting only a final classification.

## System Workflow

```text
Retinal Fundus Image
        ↓
Image Quality Assessment
        ↓
Pre-processing
        ↓
AI Screening Model
        ↓
Diabetic Retinopathy Result
        ↓
Grad-CAM Explainability
        ↓
Screening Report

