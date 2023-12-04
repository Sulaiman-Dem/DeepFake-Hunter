<div style="text-align:center">
    <!-- <h1  align="center" >Deepfake Hunter</h1> -->
    <img src="Image\Deepfake_Hunter.png" alt="Deepfake-Hunter-Logo" width="500" height="">
</div>

Dev Team:

1. Sulaiman Dem-Zerigat
   [LinkedIn](https://www.linkedin.com/in/sulaiman-dem-zerigat-43379a169/)
   [Portfolio](https://sulaiman-dem.github.io/)
2. Sung Yi
   [LinkedIn](https://www.linkedin.com/in/sung-yi-901763192/)

---

Test our AI Model here:
[App](https://huggingface.co/spaces/UW123/Deepfake_Hunter)

---

**Data Preparation: Sulaiman and Sung**

**Inspect and Explore Data (EDA): Sulaiman and Sung**

**Select and Engineer Features (Augmentation and Extraction): Sulaiman and Sung**

**Build and Train Model: Sulaiman and Sung**

**Evaluate Model: Sulaiman and Sung**

---

Important Note: This was run in Google Colab Pro using T4 GPU High-Ram.

---

## INTRODUCTION:

The primary objective of our AI model is to distinguish between authentic and fabricated audio, specifically focusing on English language audio. This application is essential for enhancing the integrity of audio content and mitigating potential harms caused by deepfake technology.

## DATASETS USED:

1. [Common Voice on Mozilla](https://commonvoice.mozilla.org/en/datasets)
   <br />
   <p>We believe that large, publicly available voice datasets will foster innovation and healthy commercial competition in machine-learning based speech technology. Common Voice’s multi-language dataset is already the largest publicly available voice dataset of its kind, but it’s not the only one.Look to this page as a reference hub for other open source voice datasets and, as Common Voice continues to grow, a home for our release updates.</p>

2. [CREMA-D: Crowd Sourced Emotional Multimodal Actors Dataset](https://www.kaggle.com/datasets/ejlok1/cremad)
   <br>
   <p>CREMA-D is a data set of 7,442 original clips from 91 actors. These clips were from 48 male and 43 female actors between the ages of 20 and 74 coming from a variety of races and ethnicities (African America, Asian, Caucasian, Hispanic, and Unspecified). Actors spoke from a selection of 12 sentences. The sentences were presented using one of six different emotions (Anger, Disgust, Fear, Happy, Neutral, and Sad) and four different emotion levels (Low, Medium, High, and Unspecified).</p>

3. [WaveFake: A Data Set to Facilitate Audio DeepFake Detection](https://github.com/RUB-SysSec/WaveFake)
   <br>
   <p>Deep generative modeling has the potential to cause significant harm to society. Recognizing this threat, a magnitude of research into detecting so-called "Deepfakes" has emerged. This research most often focuses on the image domain, while studies exploring generated audio signals have - so far - been neglected. In this paper, we aim to narrow this gap. We present a novel data set, for which we collected ten sample sets from six different network architectures, spanning two languages. We analyze the frequency statistics comprehensively, discovering subtle differences between the architectures, specifically among the higher frequencies. Additionally, to facilitate further development of detection methods, we implemented three different classifiers adopted from the signal processing community to give practitioners a baseline to compare against. In a first evaluation, we already discovered significant trade-offs between the different approaches. Neural network-based approaches performed better on average, but more traditional models proved to be more robust.</p>

4. [DEEP-VOICE: DeepFake Voice Recognition on Kaggle](https://www.kaggle.com/datasets/birdy654/deep-voice-deepfake-voice-recognition/data)
   <br>
   <p>There are growing implications surrounding generative AI in the speech domain that enable voice cloning and real-time voice conversion from one individual to another. This technology poses a significant ethical threat and could lead to breaches of privacy and misrepresentation, thus there is an urgent need for real-time detection of AI-generated speech for DeepFake Voice Conversion.To address the above emerging issues, we are introducing the DEEP-VOICE dataset. DEEP-VOICE is comprised of real human speech from eight well-known figures and their speech converted to one another using Retrieval-based Voice Conversion.</p>

5. ['In-the-Wild' Audio Deepfake Data](https://deepfake-demo.aisec.fraunhofer.de/in_the_wild)
   <br>
   <p>We present a dataset of audio deepfakes (and corresponding benign audio) for a set of politicians and other public figures, collected from publicly available sources such as social networks and video streaming platforms. For n = 58 celebrities and politicians, we collect both bona-fide and spoofed audio. In total, we collect 20.8 hours of bona-fide and 17.2 hours of spoofed audio. On average, there are 23 minutes of bona-fide and 18 minutes of spoofed audio per speaker.The dataset is intended to be used for evaluating 'Deepfake Detection' or anti-spoof machine-learning models. It is especially useful to judge a model's capability to generalize to realistic, in-the-wild audio samples.</p>

## TECHNOLOGIES:

1. [Librosa](https://librosa.org)
2. [Numpy](https://numpy.org/)
3. [Pandas](https://pandas.pydata.org/)
4. [Seaborn](https://seaborn.pydata.org/)
5. [Plotly](https://plotly.com/)
6. [Tensorflow/Keras](https://www.tensorflow.org/)
7. [Scikit-Learn](https://scikit-learn.org/stable/)
8. [Kaggle](https://www.kaggle.com/)
9. [Streamlit](https://streamlit.io/)
10. [Hugging Face](https://huggingface.co/)

## SETUP INSTRUCTIONS:

First download all our files from the github. Then open [Google Colab](https://colab.google) and make a google account if you don't have one already. Click on "Open Colab" then it will ask you to open a notebook. Click upload and individually upload each file you want to open.

> Note: Be very careful of file pathing because you would have to change the path files in the code to your respected workspace.

## Future Improvements:

This model was made with [tensorflow sequential API](https://www.tensorflow.org/api_docs/python/tf/keras/Sequential). Our next version will be use [pytorch](https://pytorch.org) instead to have more customization of the model.
