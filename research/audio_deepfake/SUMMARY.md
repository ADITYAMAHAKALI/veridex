# Research Summary: AI Voice Detection

## [A Practical Synthesis of Detecting AI-Generated Textual, Visual, and Audio Content](https://arxiv.org/abs/2504.02898)

**Abstract**:
Advances in AI-generated content have led to wide adoption of large language models, diffusion-based visual generators, and synthetic audio tools. However, these developments raise critical concerns about misinformation, copyright infringement, security threats, and the erosion of public trust. In this paper, we explore an extensive range of methods designed to detect and mitigate AI-generated textual, visual, and audio content. We begin by discussing motivations and potential impacts associated with AI-based content generation, including real-world risks and ethical dilemmas. We then outline detection techniques spanning observation-based strategies, linguistic and statistical analysis, model-based pipelines, watermarking and fingerprinting, as well as emergent ensemble approaches. We also present new perspectives on robustness, adaptation to rapidly improving generative architectures, and the critical role of human-in-the-loop verification. By surveying state-of-the-art research and highlighting case studies in academic, journalistic, legal, and industrial contexts, this paper aims to inform robust solutions and policymaking. We conclude by discussing open challenges, including adversarial transformations, domain generalization, and ethical concerns, thereby offering a holistic guide for researchers, practitioners, and regulators to preserve content authenticity in the face of increasingly sophisticated AI-generated media.

**PDF**: [Local Copy](2504.02898.pdf)

---

## [Cross-Technology Generalization in Synthesized Speech Detection: Evaluating AST Models with Modern Voice Generators](https://arxiv.org/abs/2503.22503)

**Abstract**:
This paper evaluates the Audio Spectrogram Transformer (AST) architecture for synthesized speech detection, with focus on generalization across modern voice generation technologies. Using differentiated augmentation strategies, the model achieves 0.91% EER overall when tested against ElevenLabs, NotebookLM, and Minimax AI voice generators. Notably, after training with only 102 samples from a single technology, the model demonstrates strong cross-technology generalization, achieving 3.3% EER on completely unseen voice generators. This work establishes benchmarks for rapid adaptation to emerging synthesis technologies and provides evidence that transformer-based architectures can identify common artifacts across different neural voice synthesis methods, contributing to more robust speech verification systems.

**PDF**: [Local Copy](2503.22503.pdf)

---

## [Voice Cloning: Comprehensive Survey](https://arxiv.org/abs/2505.00579)

**Abstract**:
Voice Cloning has rapidly advanced in today&#39;s digital world, with many researchers and corporations working to improve these algorithms for various applications. This article aims to establish a standardized terminology for voice cloning and explore its different variations. It will cover speaker adaptation as the fundamental concept and then delve deeper into topics such as few-shot, zero-shot, and multilingual TTS within that context. Finally, we will explore the evaluation metrics commonly used in voice cloning research and related datasets. This survey compiles the available voice cloning algorithms to encourage research toward its generation and detection to limit its misuse.

**PDF**: [Local Copy](2505.00579.pdf)

---

## [Detect All-Type Deepfake Audio: Wavelet Prompt Tuning for Enhanced Auditory Perception](https://arxiv.org/abs/2504.06753)

**Abstract**:
The rapid advancement of audio generation technologies has escalated the risks of malicious deepfake audio across speech, sound, singing voice, and music, threatening multimedia security and trust. While existing countermeasures (CMs) perform well in single-type audio deepfake detection (ADD), their performance declines in cross-type scenarios. This paper is dedicated to studying the alltype ADD task. We are the first to comprehensively establish an all-type ADD benchmark to evaluate current CMs, incorporating cross-type deepfake detection across speech, sound, singing voice, and music. Then, we introduce the prompt tuning self-supervised learning (PT-SSL) training paradigm, which optimizes SSL frontend by learning specialized prompt tokens for ADD, requiring 458x fewer trainable parameters than fine-tuning (FT). Considering the auditory perception of different audio types,we propose the wavelet prompt tuning (WPT)-SSL method to capture type-invariant auditory deepfake information from the frequency domain without requiring additional training parameters, thereby enhancing performance over FT in the all-type ADD task. To achieve an universally CM, we utilize all types of deepfake audio for co-training. Experimental results demonstrate that WPT-XLSR-AASIST achieved the best performance, with an average EER of 3.58% across all evaluation sets. The code is available online.

**PDF**: [Local Copy](2504.06753.pdf)

---

## [Does Audio Deepfake Detection Generalize?](https://arxiv.org/abs/2203.16263)

**Abstract**:
Current text-to-speech algorithms produce realistic fakes of human voices, making deepfake detection a much-needed area of research. While researchers have presented various techniques for detecting audio spoofs, it is often unclear exactly why these architectures are successful: Preprocessing steps, hyperparameter settings, and the degree of fine-tuning are not consistent across related work. Which factors contribute to success, and which are accidental? In this work, we address this problem: We systematize audio spoofing detection by re-implementing and uniformly evaluating architectures from related work. We identify overarching features for successful audio deepfake detection, such as using cqtspec or logspec features instead of melspec features, which improves performance by 37% EER on average, all other factors constant. Additionally, we evaluate generalization capabilities: We collect and publish a new dataset consisting of 37.9 hours of found audio recordings of celebrities and politicians, of which 17.2 hours are deepfakes. We find that related work performs poorly on such real-world data (performance degradation of up to one thousand percent). This may suggest that the community has tailored its solutions too closely to the prevailing ASVSpoof benchmark and that deepfakes are much harder to detect outside the lab than previously thought.

**PDF**: [Local Copy](2203.16263.pdf)

---

## [Audio Deepfake Detection: A Survey](https://arxiv.org/abs/2308.14970)

**Abstract**:
Audio deepfake detection is an emerging active topic. A growing number of literatures have aimed to study deepfake detection algorithms and achieved effective performance, the problem of which is far from being solved. Although there are some review literatures, there has been no comprehensive survey that provides researchers with a systematic overview of these developments with a unified evaluation. Accordingly, in this survey paper, we first highlight the key differences across various types of deepfake audio, then outline and analyse competitions, datasets, features, classifications, and evaluation of state-of-the-art approaches. For each aspect, the basic techniques, advanced developments and major challenges are discussed. In addition, we perform a unified comparison of representative features and classifiers on ASVspoof 2021, ADD 2023 and In-the-Wild datasets for audio deepfake detection, respectively. The survey shows that future research should address the lack of large scale datasets in the wild, poor generalization of existing detection methods to unknown fake attacks, as well as interpretability of detection results.

**PDF**: [Local Copy](2308.14970.pdf)

---

## [SpeechFake: A Large-Scale Multilingual Speech Deepfake Dataset Incorporating Cutting-Edge Generation Methods](https://arxiv.org/abs/2507.21463)

**Abstract**:
As speech generation technology advances, the risk of misuse through deepfake audio has become a pressing concern, which underscores the critical need for robust detection systems. However, many existing speech deepfake datasets are limited in scale and diversity, making it challenging to train models that can generalize well to unseen deepfakes. To address these gaps, we introduce SpeechFake, a large-scale dataset designed specifically for speech deepfake detection. SpeechFake includes over 3 million deepfake samples, totaling more than 3,000 hours of audio, generated using 40 different speech synthesis tools. The dataset encompasses a wide range of generation techniques, including text-to-speech, voice conversion, and neural vocoder, incorporating the latest cutting-edge methods. It also provides multilingual support, spanning 46 languages. In this paper, we offer a detailed overview of the dataset&#39;s creation, composition, and statistics. We also present baseline results by training detection models on SpeechFake, demonstrating strong performance on both its own test sets and various unseen test sets. Additionally, we conduct experiments to rigorously explore how generation methods, language diversity, and speaker variation affect detection performance. We believe SpeechFake will be a valuable resource for advancing speech deepfake detection and developing more robust models for evolving generation techniques.

**PDF**: [Local Copy](2507.21463.pdf)

---

## [Every Breath You Don&#39;t Take: Deepfake Speech Detection Using Breath](https://arxiv.org/abs/2404.15143)

**Abstract**:
Deepfake speech represents a real and growing threat to systems and society. Many detectors have been created to aid in defense against speech deepfakes. While these detectors implement myriad methodologies, many rely on low-level fragments of the speech generation process. We hypothesize that breath, a higher-level part of speech, is a key component of natural speech and thus improper generation in deepfake speech is a performant discriminator. To evaluate this, we create a breath detector and leverage this against a custom dataset of online news article audio to discriminate between real/deepfake speech. Additionally, we make this custom dataset publicly available to facilitate comparison for future work. Applying our simple breath detector as a deepfake speech discriminator on in-the-wild samples allows for accurate classification (perfect 1.0 AUPRC and 0.0 EER on test data) across 33.6 hours of audio. We compare our model with the state-of-the-art SSL-wav2vec model and show that this complex deep learning model completely fails to classify the same in-the-wild samples (0.72 AUPRC and 0.99 EER).

**PDF**: [Local Copy](2404.15143.pdf)

---

## [RawBMamba: End-to-End Bidirectional State Space Model for Audio Deepfake Detection](https://arxiv.org/abs/2406.06086)

**Abstract**:
Fake artefacts for discriminating between bonafide and fake audio can exist in both short- and long-range segments. Therefore, combining local and global feature information can effectively discriminate between bonafide and fake audio. This paper proposes an end-to-end bidirectional state space model, named RawBMamba, to capture both short- and long-range discriminative information for audio deepfake detection. Specifically, we use sinc Layer and multiple convolutional layers to capture short-range features, and then design a bidirectional Mamba to address Mamba&#39;s unidirectional modelling problem and further capture long-range feature information. Moreover, we develop a bidirectional fusion module to integrate embeddings, enhancing audio context representation and combining short- and long-range information. The results show that our proposed RawBMamba achieves a 34.1\% improvement over Rawformer on ASVspoof2021 LA dataset, and demonstrates competitive performance on other datasets.

**PDF**: [Local Copy](2406.06086.pdf)

---

## [Retrieval-Augmented Audio Deepfake Detection](https://arxiv.org/abs/2404.13892)

**Abstract**:
With recent advances in speech synthesis including text-to-speech (TTS) and voice conversion (VC) systems enabling the generation of ultra-realistic audio deepfakes, there is growing concern about their potential misuse. However, most deepfake (DF) detection methods rely solely on the fuzzy knowledge learned by a single model, resulting in performance bottlenecks and transparency issues. Inspired by retrieval-augmented generation (RAG), we propose a retrieval-augmented detection (RAD) framework that augments test samples with similar retrieved samples for enhanced detection. We also extend the multi-fusion attentive classifier to integrate it with our proposed RAD framework. Extensive experiments show the superior performance of the proposed RAD framework over baseline methods, achieving state-of-the-art results on the ASVspoof 2021 DF set and competitive results on the 2019 and 2021 LA sets. Further sample analysis indicates that the retriever consistently retrieves samples mostly from the same speaker with acoustic characteristics highly consistent with the query audio, thereby improving detection performance.

**PDF**: [Local Copy](2404.13892.pdf)

---
