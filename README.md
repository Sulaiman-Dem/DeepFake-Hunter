# Deep-Fake-Audio-Detection

Team members:

1. Sulaiman Dem-Zerigat
2. Sung Yi
3. Brain Sternfeld

## Description:

The primary objective of our AI model is to distinguish between authentic and fabricated audios, specifically focusing on English language audio. This application is essential for enhancing the integrity of audio content and mitigating potential harms caused by deepfake technology.

## Implementation:

Our AI model will be constructed as a neural network, utilizing supervised learning initially. We will feed it labeled data – a mix of both genuine and tampered audios – to allow the model to learn the characteristics that differentiate the two. After the initial phase, we plan to transition to reinforcement learning. This approach will enable the model to refine its predictions based on feedback, optimizing its performance over time. Our primary choice of framework for implementing this model is PyTorch due to its flexibility and efficiency in developing deep learning models. However, we will consider TensorFlow as an alternative if we encounter problems that can't be readily addressed in PyTorch.

## Issues:

The nature of this project presents a few challenges. First, as with any deep learning model, training the network will be time-intensive. We'll need to factor this into our project timeline. Second, there's the issue of potential bias in the training data. To correct for this, we will implement a weighted node value system. This approach will allow us to assign greater importance to certain audios we definitively know to be real or fake. However, we must be cautious in adjusting these weights. Overcompensation could skew the model, lowering variance and potentially leading to overfitting. We'll need to keep a close eye on the model's performance metrics to ensure balance.

## Workload:

The project will be a team effort, with all members learning and contributing to the development of the AI. Rather than compartmentalizing tasks, we believe that a collaborative approach will yield the best results. By working together, we can collectively troubleshoot problems, share insights, and enhance our understanding of AI applications. Each team member's input will be critical in shaping the model, from data preprocessing to final testing and evaluation.

## Future Scope:

While our initial focus is on English audio, we anticipate our model could be adapted for other languages in the future. Furthermore, as deepfake technology continues to evolve, our model's ability to detect falsified audios could have significant implications for the fields of cybersecurity, media integrity, and beyond. This project, therefore, holds substantial potential for future development and application.
