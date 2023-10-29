# Deep-Fake-Audio-Detection

! Important Note: This was run in google colab pro using A100 GPU.

Team members:

1. Sulaiman Dem-Zerigat
2. Sung Yi

## Description:

The primary objective of our AI model is to distinguish between authentic and fabricated audio, specifically focusing on English language audio. This application is essential for enhancing the integrity of audio content and mitigating potential harms caused by deepfake technology.

## Implementation:

Our AI model will be constructed as a neural network, utilizing supervised learning initially. We will feed it labelled data – a mix of both genuine and tampered audio – to allow the model to learn the characteristics that differentiate the two. After the initial phase, we plan to transition to reinforcement learning. This approach will enable the model to refine its predictions based on feedback, optimizing its performance over time. Our primary choice of framework for implementing this model is Tensorflow due to its simplicity with Sequential API. Since this will be our first-ever model.

## Issues:

The nature of this project presents a few challenges. First, as with any deep learning model, training the network will be time-intensive. We'll need to factor this into our project timeline. Second, there's the issue of potential bias in the training data. To correct this, we will implement a weighted node value system. This approach will allow us to assign greater importance to certain audio we definitively know to be real or fake. However, we must be cautious in adjusting these weights. Overcompensation could skew the model, lowering variance and potentially leading to overfitting. We'll need to keep a close eye on the model's performance metrics to ensure balance.

## Workload:

The project will be a team effort, with all members learning and contributing to the development of the AI. Rather than compartmentalizing tasks, we believe that a collaborative approach will yield the best results. By working together, we can collectively troubleshoot problems, share insights, and enhance our understanding of AI applications. Each team member's input will be critical in shaping the model, from data preprocessing to final testing and evaluation.

## Future Scope:

While our initial focus is on English audio, we anticipate our model could be adapted for other languages in the future. Furthermore, as deepfake technology continues to evolve, our model's ability to detect falsified audio could have significant implications for the fields of cybersecurity, media integrity, and beyond. This project, therefore, holds substantial potential for future development and application.
