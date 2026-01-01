# **Comprehensive Technical Analysis for the Development of a Unified Generative AI Detection Library**

## **1. Introduction: The Epistemological Crisis of Synthetic Media**

The rapid ascendancy of Generative Artificial Intelligence (GenAI) has precipitated a fundamental shift in the digital information ecosystem, transitioning from a regime of scarcity—where content creation required human cognitive effort—to a regime of abundance, where Large Language Models (LLMs) and Denoising Diffusion Probabilistic Models (DDPMs) can synthesize localized realities at scale. For a developer tasked with constructing a detection library, this necessitates a move beyond simple binary classification into a multi-modal forensic architecture. The library must not merely detect "fake" content but must discern the specific generative provenance—distinguishing between a transformer-based text hallucination, a diffusion-based image synthesis, and a neural-vocoder-generated audio clip.

The current landscape of detection is bifurcated into two dominant methodologies: **Passive Detection**, which relies on identifying statistical anomalies, spectral artifacts, and semantic inconsistencies inherent to the generation process; and **Active Provenance**, which involves the cryptographic verification of content history via standards like C2PA or the decoding of imperceptible watermarks embedded during generation.

This report provides an exhaustive technical analysis of the state-of-the-art (SOTA) techniques available in 2024-2025 for identifying AI-generated content across text, image, video, and audio modalities. It is structured to serve as a foundational architectural document for the "GenAI Detect" library, detailing the theoretical underpinnings, algorithmic implementations, and dataset benchmarks required to build a robust, production-grade system.

### **1.1 The Generative Architectures and Their Fingerprints**

To detect the artifact, one must understand the architect. The detection library must target the specific mathematical signatures left by the three primary generative architectures dominating the current market:

| Architecture | Modality | Generative Mechanism | Primary Forensic Fingerprint |
| :---- | :---- | :---- | :---- |
| **Transformers** (GPT-4, Llama 3\) | Text, Code | Autoregressive Next-Token Prediction | Low Perplexity, Low Burstiness, Probability Curvature |
| **Diffusion Models** (Stable Diffusion, Flux) | Image, Video | Iterative Denoising (Gaussian $\\to$ Data) | High-Frequency Spectral Decay, Reconstruction Error (DIRE) |
| **Neural Vocoders** (HiFi-GAN, WaveNet) | Audio | Mel-Spectrogram Inversion | Phase Discontinuities, Vocoder Artifacts in High Frequencies |
| **GANs** (StyleGAN) | Image (Legacy) | Adversarial Min-Max Game | Checkerboard/Grid Artifacts, Up-sampling Spectral Spikes |

The following sections analyze these fingerprints in granular detail, providing the necessary roadmap for implementation within a Python-based detection framework.

## ---

**2. Text Detection: Statistical Thermodynamics and Latent Watermarking**

The detection of AI-generated text is fundamentally a problem of statistical thermodynamics applied to linguistics. Human language production is a stochastic process driven by complex, often chaotic cognitive states, resulting in high entropy. In contrast, LLMs are optimization engines designed to minimize the cross-entropy loss between the predicted token distribution and the training data. This optimization objective forces LLM outputs into a narrower, more probable region of the linguistic manifold, creating detectable statistical signatures.

### **2.1 Zero-Shot Statistical Detection Methods**

For a detection library, zero-shot methods are of paramount importance. Unlike supervised classifiers, which require constant retraining to keep pace with new model releases (e.g., the shift from GPT-3 to GPT-4), zero-shot methods leverage the invariant properties of the transformer architecture itself. They do not learn "what AI text looks like" but rather "how AI models process text."

#### **2.1.1 Perplexity and Burstiness: The Baseline Metrics**

The foundational metrics for any text detection module are **Perplexity (PPL)** and **Burstiness**. These metrics quantify the "surprise" and "variance" of a text sequence, respectively.

**Perplexity** is defined as the exponentiated average negative log-likelihood of a sequence. For a sequence of tokens $X \= (x\_1, x\_2,..., x\_t)$, the perplexity measured by a model $\\theta$ is:

$$PPL(X) \= \\exp \\left( \-\\frac{1}{t} \\sum\_{i=1}^{t} \\log P\_\\theta(x\_i | x\_{\<i}) \\right)$$  
In practical implementation using the Hugging Face transformers library, this involves loading a causal language model (like GPT-2 or Llama-2-7b), feeding the input text, and calculating the loss. Research consistently demonstrates that AI-generated text exhibits significantly lower perplexity than human-authored text.1 Humans, unconstrained by the objective of maximizing probability, frequently choose "sub-optimal" words that add color, nuance, or specific intent, thereby raising the perplexity.2

**Burstiness** complements perplexity by measuring the variation of perplexity over time (i.e., across sentences). Human writing is characterized by high burstiness: a complex, high-perplexity sentence is often followed by a short, low-perplexity clarification. LLMs, aiming for consistent quality, tend to generate text with a uniform perplexity profile, resulting in low burstiness scores.3

Library Implementation Strategy:  
The detection library should include a StatisticalAnalyzer class that utilizes a lightweight model (e.g., gpt2-medium) to compute these metrics.

* **Input:** Raw text string.  
* **Process:** Tokenize text \-\> Split into sentences \-\> Compute PPL per sentence.  
* **Output:** Mean PPL (Entropy) and Standard Deviation of PPL (Burstiness).  
* **Thresholding:** A logistic regression classifier can be trained on these two scalars to provide a preliminary "Human vs. AI" score.4

#### **2.1.2 Binoculars: Contrastive Perplexity Analysis**

While raw perplexity is useful, it suffers from domain bias; a human writing a legal contract (highly structured) might have lower perplexity than an AI writing a surreal poem. To mitigate this, the **Binoculars** method introduces a contrastive scoring mechanism that is currently state-of-the-art for zero-shot detection.5

Theoretical Mechanism:  
Binoculars relies on the interplay between two models: a Performer (a strong model, e.g., Falcon-7B) and an Observer (a slightly weaker or identical model, e.g., Falcon-7B-Instruct). The core intuition is that machine-generated text is "surprising" to the Observer in a very specific way that correlates with how "confident" the Performer was in generating it.  
The Binoculars score $\\mathcal{B}$ is calculated as the ratio of the log-perplexity of the two models:

$$\\mathcal{B} \= \\frac{\\log PPL\_{Observer}(X)}{\\log PPL\_{Performer}(X)}$$  
Performance and Robustness:  
Empirical evaluations show that Binoculars achieves over 90% detection accuracy on ChatGPT and Llama outputs with a False Positive Rate (FPR) of only 0.01%.6 Crucially, it is domain-agnostic, performing equally well on creative writing, academic papers, and code, which makes it an ideal candidate for a general-purpose library.  
**Implementation Details:**

* **Dependencies:** Requires transformers and substantial VRAM to load the Performer and Observer models (though quantization can mitigate this).  
* **Code Structure:** The library should implement a BinocularsDetector class.  
  * \_\_init\_\_: Load the two models (or a single model acting as both if using different prompting strategies).  
  * score(text): Compute the perplexity ratio.  
  * predict(text): Compare the score against a pre-calibrated threshold (typically derived from the ROC curve on a validation set like DetectRL).5

#### **2.1.3 Fast-DetectGPT: Conditional Probability Curvature**

A second critical zero-shot method is **Fast-DetectGPT**, which offers a computationally efficient alternative to the original DetectGPT.

The Curvature Hypothesis:  
DetectGPT posited that AI-generated text resides in a region of negative log-probability curvature. If we take an AI-generated passage and apply random perturbations (e.g., synonym replacement), the probability of the perturbed text under the generating model drops sharply. Conversely, human text, which is already in a region of higher entropy (lower probability), does not suffer as significant a drop when perturbed.7  
The "Fast" Optimization:  
The original method required generating hundreds of perturbed samples, which was prohibitively slow for real-time applications. Fast-DetectGPT approximates this curvature analytically using the conditional probability distributions of the tokens, eliminating the sampling step. This results in a speedup of approximately 340x, enabling high-throughput detection suitable for API integration.7  
Library Integration:  
This method is particularly valuable for the library because it can utilize the same base model loaded for Binoculars, maximizing resource efficiency. The FastDetectGPT module would analyze the local curvature of the probability surface around the input text to determine authorship probability.10

### **2.2 Supervised Deep Learning Detectors**

While zero-shot methods provide robustness, supervised models fine-tuned on specific datasets can offer higher accuracy for known threat models (e.g., detecting essays written by GPT-3.5).

#### **2.2.1 RoBERTa-Base OpenAI Detector**

The **RoBERTa-base-openai-detector** represents the baseline for supervised detection. It treats the problem as a binary sequence classification task.11

Architecture:  
The model utilizes a RoBERTa encoder (a robustly optimized BERT variant) initialized with pre-trained weights. A classification head (a dense linear layer with softmax) is added on top of the \`\` token output. The model is fine-tuned on a dataset of human-written text and text generated by models like GPT-2.  
Limitations and Concept Drift:  
A major limitation of supervised detectors is concept drift. A model trained to detect GPT-2 will struggle to detect GPT-4 or Claude 3, as the newer models generate text that is statistically closer to human distribution. Furthermore, supervised models are prone to overfitting to specific domains (e.g., Reddit comments) and fail when tested on out-of-distribution data (e.g., medical reports).13 The library should include these models but flag them as "version-specific" detectors.

#### **2.2.2 The Move to Ternary Classification**

Recent research challenges the binary "Human vs. AI" paradigm. With the prevalence of AI-assisted tools (Grammarly, Notion AI, human-in-the-loop editing), a strictly binary classifier produces high false positive rates for hybrid content.  
Recommendation: The library should adopt a Ternary Classification Scheme:

1. **Human:** High burstiness, high perplexity, low Binoculars score.  
2. **AI-Generated:** Low burstiness, low perplexity, high probability curvature.  
3. **Undecided / Hybrid:** Text that exhibits conflicting signals (e.g., human sentence structure with AI-like vocabulary selection). This category is crucial for user trust and explainability.14

### **2.3 Active Detection: The SynthID Watermarking Protocol**

The most robust form of detection avoids probabilistic guessing entirely by embedding a deterministic signal into the text generation process. **SynthID Text**, developed by Google DeepMind, is the standard for this approach.

Mechanism: Logits Processing with g-functions:  
SynthID operates at the logits level—the unnormalized probability scores assigned to each token in the vocabulary before sampling.

1. **Tournament Selection:** The algorithm uses a pseudo-random function $g$, seeded by a secret key and the preceding context (tokens), to assign a "score" to each possible next token.  
2. **Biasing:** It subtly boosts the probability of tokens that have a high $g$-score.  
3. **Sampling:** The model samples the next token using these modified probabilities.  
4. **Result:** The generated text contains a statistical bias towards tokens with high $g$-scores. This bias is imperceptible to humans (preserving text quality) but can be detected by a verifier model that knows the secret key and the $g$-function.15

Detection via Bayesian Hypothesis Testing:  
The detector calculates the mean $g$-score of the tokens in the document. If the mean is significantly higher than the expected mean for random text, the document is confirmed as watermarked. This allows for detection even with short text snippets (as few as three sentences) and provides a quantifiable confidence level.15  
Library Implementation:  
SynthID Text has been open-sourced and integrated into the Hugging Face transformers library (v4.46.0+). The detection library should wrap the SynthIDTextWatermarkDetector class, allowing users to verify content generated by models that utilize this watermarking scheme (e.g., Gemini, Imagen-text variants).15

### **2.4 Text Datasets and Benchmarks**

To validate the efficacy of the text detection module, the library must include or reference standardized benchmarks.

| Dataset / Benchmark | Description | Key Features | Utility for Library | Source |
| :---- | :---- | :---- | :---- | :---- |
| **DetectRL** | Adversarial benchmark using RL to modify text. | Contains "humanified" AI text (misspellings, paraphrasing) to stress-test detectors. | Evaluating robustness against evasion attacks. | 17 |
| **SHIELD** | Hardness-aware benchmark. | Evaluates detector stability across diverse domains (legal, creative, technical). | Mitigating false positives in low-perplexity human text. | 18 |
| **Voight-Kampff 2024** | Authorship verification challenge. | Focuses on perplexity and burstiness validation across varied generators. | Validating statistical metrics. | 4 |
| **M4 Dataset** | Massive Multi-model corpus. | Text from varying model sizes (bloom-7b to gpt-4) mixed with human text. | General-purpose training/testing. | 10 |

## ---

**3. Image Detection: Spectral Analysis and Latent Reconstruction**

The detection of AI-generated images involves identifying artifacts in the spatial domain (pixel inconsistencies) and the frequency domain (spectral anomalies). While early GANs left obvious visual artifacts (e.g., asymmetrical eyes), modern Diffusion Models (DMs) like Stable Diffusion 3.5 and Midjourney v6 produce photorealistic outputs that require deep forensic analysis.

### **3.1 Frequency Domain Forensics**

The frequency domain is often where the "reality" of an image is authenticated. Real photographs, captured by physical sensors, undergo a specific pipeline: photon capture \-\> Color Filter Array (CFA) interpolation (demosaicing) \-\> compression. This process leaves a distinct spectral signature. Generative models, which synthesize images from noise, lack this physical provenance.

#### **3.1.1 Spectral Roll-off and Beta Statistics**

Diffusion models generate images by iteratively denoising a Gaussian noise field. This process tends to "smooth" the high-frequency components differently than a physical lens.

* **Analysis:** By applying a Discrete Fourier Transform (DFT) or Discrete Cosine Transform (DCT) to the image, one can visualize the power spectrum.  
* **The Artifact:** Real images typically exhibit a power law decay in frequency energy ($1/f^\\alpha$). Diffusion images often show a steeper "roll-off" at high frequencies or anomalous energy concentrations in specific bands due to the upsampling layers in the U-Net architecture.19  
* **Library Implementation:** The library should use opencv-python to compute the 2D DFT (cv2.dft). A SpectralAnalyzer module can then compute the azimuthal average of the power spectrum and compare the decay rate ($\\beta$) against thresholds derived from real image datasets (like MS-COCO).19

#### **3.1.2 FreqCross: Multi-Modal Fusion**

State-of-the-art detectors like **FreqCross** explicitly fuse spatial and frequency features to improve robustness.

* **Architecture:** The model uses a two-branch architecture. One branch (e.g., ResNet-18) processes the RGB image to capture spatial semantics. The second branch processes the frequency spectrum (often represented as a heatmap or log-magnitude plot). The features are concatenated and passed to a classifier.21  
* **Relevance:** This approach is particularly effective against images that have been resized or compressed, actions that often destroy subtle pixel-level artifacts but leave spectral traces intact.22

### **3.2 Diffusion Reconstruction Error (DIRE)**

**DIRE** (DIffusion Reconstruction Error) is a hypothesis-driven method that leverages the generative model itself as a forensic tool.

The Reconstruction Hypothesis:  
The core insight is that a diffusion model is an optimal compressor for images it could have generated. If an image was generated by Stable Diffusion, it lies within the model's learned latent manifold. Therefore, if we add noise to the image and then ask Stable Diffusion to "denoise" (reconstruct) it, the reconstruction should be highly accurate.  
Conversely, a real photograph contains complex, chaotic high-frequency details (sensor noise, grain, infinite depth of field) that the diffusion model did not learn. Thus, the reconstruction of a real image will be lossy.23  
**Algorithm:**

1. **Input:** Image $x\_0$.  
2. **Forward Diffusion:** Add Gaussian noise to reach timestep $t$ (e.g., $t=100$): $x\_t \= \\sqrt{\\bar{\\alpha}\_t} x\_0 \+ \\sqrt{1 \- \\bar{\\alpha}\_t} \\epsilon$.  
3. **Reverse Diffusion:** Use the pre-trained diffusion model (e.g., Stable Diffusion v1.5) to denoise $x\_t$ back to an estimate $x\_0'$.  
4. **Error Map:** Compute the absolute difference: DIRE $= |x\_0 \- x\_0'|$.  
5. **Classification:** A CNN (like ResNet-50) is trained on these DIRE maps. "Fake" images result in low-energy DIRE maps (black/empty), while "Real" images result in high-energy maps highlighting the lost details.24

Library Integration:  
Implementing DIRE requires the diffusers library to load the specific checkpoint (e.g., runwayml/stable-diffusion-v1-5). The library should allow users to specify the "probe" model. Note that this method is computationally expensive (requires GPU inference) compared to spectral analysis but offers higher accuracy.25

### **3.3 Noise Fingerprinting (Noiseprint)**

To determine *which* model generated an image (Provenance), **Noiseprint** analysis is effective.

* **Concept:** Similar to Photo-Response Non-Uniformity (PRNU) in physical cameras—where silicon imperfections create a unique noise pattern—generative models leave "model fingerprints" in the noise residuals due to their specific architecture (kernel sizes, stride, upsampling method).26  
* **Implementation:** The library can utilize a pre-trained Siamese network (CNN) that extracts the noise residual while suppressing scene content. By comparing the extracted fingerprint against a database of known model fingerprints (Midjourney, DALL-E, SDXL), the specific generator can be identified.27

### **3.4 Image Watermarking: Latent and Post-Hoc**

A complete library must support the decoding of watermarks, which serve as the "ground truth" when present.

#### **3.4.1 Stable Signature (Latent Watermarking)**

Developed by Meta, **Stable Signature** integrates watermarking into the training of the diffusion model itself.

* **Method:** The decoder of the Latent Diffusion Model (LDM) is fine-tuned to embed a fixed binary signature into every image it reconstructs from latent space.  
* **Resilience:** Because the watermark is part of the image formation process, it is robust to cropping (down to 10%), rotation, and mild JPEG compression. The detection library can use the provided extractor network to recover this signature.28

#### **3.4.2 Invisible-Watermark (DWT/DCT)**

The invisible-watermark Python library is widely used in the open-source community (e.g., by the AUTOMATIC1111 web UI).

* **Algorithm:** It uses the **Discrete Wavelet Transform (DWT)** to decompose the image into frequency bands (LL, LH, HL, HH). The watermark bits are embedded into the coefficients of the mid-frequency bands, balancing invisibility and robustness.  
* **Library Code:** The detection library should wrap this dependency:  
  Python  
  from imwatermark import WatermarkDecoder  
  def decode\_dwt(image\_path):  
      bgr \= cv2.imread(image\_path)  
      decoder \= WatermarkDecoder('bytes', 32)  
      watermark \= decoder.decode(bgr, 'dwtDct')  
      return watermark

  This allows detection of standard tags often embedded by Stable Diffusion pipelines.30

### **3.5 Image Datasets and Benchmarks**

| Dataset | Modality | Size | Description | Source |
| :---- | :---- | :---- | :---- | :---- |
| **CIFAKE** | Image | 120k | Balanced dataset of CIFAR-10 real images vs. Stable Diffusion generated variants. Ideal for training lightweight classifiers. | 32 |
| **CNNDetection** | Image | Various | A benchmark by Wang et al. showing that detectors trained on ProGAN generalize to other GANs and some DMs. | 33 |
| **DiffusionForensics** | Image | Multi-model | Specifically curated for evaluating the DIRE method, containing images from ADM, Glide, and SD. | 24 |
| **VID-AID** | Video/Image | 10k Videos | Although a video dataset, its frames provide a massive source of diverse AI imagery from 9 different T2V models. | 34 |

## ---

**4. Video Detection: Temporal Dynamics and Biological Signals**

Video detection is exponentially more complex than image detection due to the temporal dimension. However, this complexity also offers new vectors for detection: temporal inconsistency and the absence of biological signals.

### **4.1 Physiological Signal Analysis (rPPG)**

The most robust "liveness" detection method relies on **Remote Photoplethysmography (rPPG)**.

* **Biological Basis:** Every time a human heart beats, the volume of blood in the facial vessels changes, causing a subtle variation in skin color (primarily in the green channel due to hemoglobin absorption). This signal is invisible to the naked eye but detectable by standard RGB cameras.  
* **The AI Flaw:** Current Deepfake models (like DeepFaceLab, SimSwap) and T2V models (Sora, Gen-2) generate pixels based on visual coherence, not biological physiology. They often enforce temporal smoothing to reduce flicker, which inadvertently "scrubs" the rPPG signal, or they generate a chaotic signal that does not correspond to a valid heart rate (60-100 BPM).35

Library Implementation:  
The library should integrate the rPPG-Toolbox, utilizing supervised models like PhysNet or DeepPhys.

1. **Face Extraction:** Use a face detector to isolate the ROI (Region of Interest).  
2. **Signal Extraction:** Compute the spatial average of the skin pixels over time.  
3. **PSD Analysis:** Compute the Power Spectral Density. A real video will show a distinct peak in the 0.7-4 Hz range (heart rate). A deepfake will show either a flat line (noise) or random peaks.37

### **4.2 Spatiotemporal Inconsistency (I3D and Xception)**

AI video generators often struggle to maintain physical consistency over time (e.g., an object morphing shape, inconsistent lighting shadows).

I3D (Inflated 3D ConvNets):  
Unlike 2D CNNs that process frames individually, 3D CNNs process a volume of frames $(T \\times H \\times W)$. This allows the model to learn motion features.

* **Detection:** I3D models trained on datasets like FaceForensics++ can detect temporal artifacts such as "flickering" boundaries or unnatural eye blinking patterns.39  
* **DeepfakeBench Integration:** The library should wrap **DeepfakeBench**, a standardized framework that includes pre-trained implementations of Xception, EfficientNet, and I3D. This allows the library to offer a "Video Scan" function that aggregates scores from multiple frame sequences.40

### **4.3 Lip-Sync and Audio-Visual Dissonance**

Specific detectors are required for "dubbing" deepfakes (e.g., Wav2Lip), where the face is real but the mouth is manipulated to match a new audio track.

* **Artifacts:** Wav2Lip often leaves blurring artifacts around the mouth region. Furthermore, the synchronization between the phonemes (audio sounds) and visemes (mouth shapes) may be imperfect.  
* **SyncNet:** A dedicated module can measure the Audio-Visual (AV) offset. If the offset varies significantly or the confidence of sync is low, it indicates manipulation.42

### **4.4 Video Datasets**

| Dataset | Focus | Scale | Utility | Source |
| :---- | :---- | :---- | :---- | :---- |
| **FaceForensics++ (FF++)** | Face Swaps | 1000 Videos | The gold standard benchmark. Contains Deepfakes, Face2Face, FaceSwap, NeuralTextures. | 44 |
| **Deepfake Detection Challenge (DFDC)** | Robustness | 100k+ Clips | Contains diverse actors, lighting, and adversarial scenarios. Essential for testing generalization. | 45 |
| **Celeb-DF** | High Quality | 590 Videos | Designed to address the saturation of FF++, offering higher visual quality fakes to challenge detectors. | 41 |

## ---

**5. Audio Detection: Neural Vocoder Fingerprints**

Synthetic audio (Voice Cloning, TTS) is typically generated using a pipeline of **Text encoder $\\to$ Mel-spectrogram decoder $\\to$ Neural Vocoder**. The vocoder (e.g., HiFi-GAN, WaveGlow) is responsible for converting the spectrogram into a waveform, and this is where most forensic artifacts are introduced.

### **5.1 Spectro-Temporal Graph Attention (AASIST)**

**AASIST** (Audio Anti-Spoofing using Integrated Spectro-Temporal Graph Attention Networks) is the current SOTA architecture for audio deepfake detection.

* **The Problem:** Standard CNNs (2D) treat spectrograms like images, ignoring the specific relationship between time and frequency bins in audio physics.  
* **The Solution:** AASIST models the audio as a graph. Nodes represent different spectral and temporal components. Graph Attention Networks (GATs) learn the relationships between these nodes. This allows the model to detect artifacts that are non-local—for example, a high-frequency glitch that correlates with a specific phoneme transition.46  
* **Performance:** AASIST outperforms previous baselines (like RawNet2) on the ASVspoof 2019 benchmark, achieving an Equal Error Rate (EER) of under 1% in many scenarios.47

### **5.2 Foundation Models: Wav2Vec 2.0**

Just as LLMs have revolutionized text detection, Self-Supervised Learning (SSL) models have revolutionized audio detection.

* **Wav2Vec 2.0:** This model is pre-trained on thousands of hours of unlabeled audio to learn discrete speech representations. When fine-tuned for the binary classification task (Real vs. Fake), it leverages its deep prior knowledge of "what human speech sounds like" to identify anomalies.48  
* **Robustness:** Research indicates that Wav2Vec 2.0-based detectors generalize better to unseen attacks (e.g., new vocoders) than models trained from scratch, making them essential for an "in-the-wild" detection library.49

### **5.3 Benchmarks and Challenges**

* **ASVspoof Challenges:** The primary benchmark series. It includes **Logical Access (LA)** (TTS and Voice Conversion attacks) and **Physical Access (PA)** (Replay attacks). A robust library must perform well on the ASVspoof 2019 and 2021 evaluation partitions.50  
* **The ElevenLabs Challenge:** Commercial TTS models like ElevenLabs produce extremely high-fidelity audio. Detection often relies on subtle "phase discontinuities" in the higher frequencies (\>8kHz) that newer vocoders still struggle to eliminate perfectly.51

## ---

**6. Provenance and Standards: The C2PA Framework**

While probabilistic detection is vital, the industry is moving towards deterministic **Provenance**. A complete detection library must support the **Coalition for Content Provenance and Authenticity (C2PA)** standard.

### **6.1 Cryptographic Binding**

C2PA allows creators (cameras, AI models, editing software) to cryptographically bind metadata to a media file.

* **Manifests:** A C2PA manifest contains assertions: "Created by Adobe Firefly," "Edited in Photoshop," "Captured by Sony A7IV."  
* **Signatures:** The manifest is signed with a private key. The public key is registered with a trust list. This ensures the metadata has not been tampered with.

### **6.2 Library Integration (c2pa-python)**

The library should integrate the c2pa-python package to serve as the "first line of defense."

* **Workflow:**  
  1. User uploads file.  
  2. Library calls c2pa.read\_manifest(path).  
  3. **Validation:** Check if the signature is valid and traces back to a trusted root.  
  4. **Assertion Check:** Look for c2pa.actions that indicate "AI Generation" or "AI Editing."  
  5. **Result:** If a valid AI manifest is found, the probability of AI is 100%. This bypasses the need for expensive and uncertain neural network inference.52

## ---

 **7. Comparison of Detection Methodologies**

The following table summarizes the methodologies to be implemented, highlighting the trade-offs between accuracy, robustness, and computational cost.

| Modality | Methodology | Classification | Accuracy (SOTA) | Robustness | Compute Cost | Key Dependency | Source |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| **Text** | Binoculars | Zero-Shot | \>90% (low FPR) | High | Med (2x LLM Inference) | transformers | 6 |
| **Text** | SynthID | Watermark | \>99% | Med (Edit-sensitive) | Low | synthid-text | 15 |
| **Text** | Fast-DetectGPT | Statistical | High | Med | Low | transformers | 7 |
| **Image** | DIRE | Reconstruction | High | High | High (Diffusion Inference) | diffusers | 24 |
| **Image** | FreqCross | Spectral Fusion | Med-High | Med (Compression) | Low | opencv, torch | 21 |
| **Video** | rPPG (PhysNet) | Biological | High | High | Med | rppg-toolbox | 38 |
| **Video** | I3D / Xception | Spatiotemporal | High | Med (compression) | High | deepfakebench | 41 |
| **Audio** | AASIST | Graph Attention | High | Med | Med | torchaudio | 46 |
| **Audio** | Wav2Vec 2.0 | Foundation Model | SOTA | High | High | transformers | 48 |
| **All** | C2PA | Provenance | 100% (if present) | Very High | Negligible | c2pa-python | 53 |

### **8. Conclusion and Future Outlook**

The development of a GenAI detection library is a pursuit of a moving target. As generative models improve (e.g., diffusion models correcting high-frequency artifacts, LLMs mimicking burstiness), passive detection methods will face diminishing returns.

* **Short-term:** The library must rely on **ensembling**. No single metric (perplexity, DIRE, rPPG) is sufficient, but their combination creates a robust decision boundary.  
* **Long-term:** The industry is inexorably moving towards **Active Provenance**. The integration of C2PA and robust watermarking (SynthID, Stable Signature) will likely supersede statistical guessing. A forward-looking library must therefore prioritize these standards as the "gold" path, utilizing passive forensics as a fallback mechanism for legacy or non-compliant content.

By implementing the multi-modal architectures detailed in this report—specifically **Binoculars for text**, **DIRE for images**, **rPPG for video**, and **AASIST for audio**—developers can construct a detection system that is resilient, explainable, and ready for the generative age.

#### **Works cited**

1. AI Text Detection: Perplexity & Burstiness | PDF | Artificial Neural Network \- Scribd, accessed January 1, 2026, [https://www.scribd.com/document/724157806/AI-Generated-Text-Detection-Synopsis](https://www.scribd.com/document/724157806/AI-Generated-Text-Detection-Synopsis)  
2. How to Distinguish Human Text From AI Text \- Using Python \- ActiveState, accessed January 1, 2026, [https://www.activestate.com/blog/how-to-distinguish-human-from-ai-text-using-python/](https://www.activestate.com/blog/how-to-distinguish-human-from-ai-text-using-python/)  
3. AdityaRajPateriya/Generative-AI-Detector \- GitHub, accessed January 1, 2026, [https://github.com/AdityaRajPateriya/Generative-AI-Detector](https://github.com/AdityaRajPateriya/Generative-AI-Detector)  
4. AkashKobal/Generative-AI-Detection \- GitHub, accessed January 1, 2026, [https://github.com/AkashKobal/Generative-AI-Detection](https://github.com/AkashKobal/Generative-AI-Detection)  
5. \[ICML 2024\] Binoculars: Zero-Shot Detection of LLM-Generated Text \- GitHub, accessed January 1, 2026, [https://github.com/ahans30/Binoculars](https://github.com/ahans30/Binoculars)  
6. Spotting LLMs With Binoculars: Zero-Shot Detection of Machine-Generated Text \- arXiv, accessed January 1, 2026, [https://arxiv.org/abs/2401.12070](https://arxiv.org/abs/2401.12070)  
7. fast-detect-gpt/supervised.sh at main \- GitHub, accessed January 1, 2026, [https://github.com/baoguangsheng/fast-detect-gpt/blob/main/supervised.sh](https://github.com/baoguangsheng/fast-detect-gpt/blob/main/supervised.sh)  
8. baoguangsheng/fast-detect-gpt: Code base for ICLR 2024 ... \- GitHub, accessed January 1, 2026, [https://github.com/baoguangsheng/fast-detect-gpt](https://github.com/baoguangsheng/fast-detect-gpt)  
9. fast-detect-gpt/scripts/local\_infer.py at main \- GitHub, accessed January 1, 2026, [https://github.com/baoguangsheng/fast-detect-gpt/blob/main/scripts/local\_infer.py](https://github.com/baoguangsheng/fast-detect-gpt/blob/main/scripts/local_infer.py)  
10. fast-detect-gpt/main.sh at main \- GitHub, accessed January 1, 2026, [https://github.com/baoguangsheng/fast-detect-gpt/blob/main/main.sh](https://github.com/baoguangsheng/fast-detect-gpt/blob/main/main.sh)  
11. openai-community/roberta-base-openai-detector \- Hugging Face, accessed January 1, 2026, [https://huggingface.co/openai-community/roberta-base-openai-detector](https://huggingface.co/openai-community/roberta-base-openai-detector)  
12. FacebookAI/roberta-base \- Hugging Face, accessed January 1, 2026, [https://huggingface.co/FacebookAI/roberta-base](https://huggingface.co/FacebookAI/roberta-base)  
13. YDSP Report \- Singapore \- DSTA, accessed January 1, 2026, [https://www.dsta.gov.sg/staticfile/ydsp/projects/files/reports/YDSP%20Report\_Evaluation%20of%20Artificial%20Intelligence%20Text%20Detection%20Models.pdf](https://www.dsta.gov.sg/staticfile/ydsp/projects/files/reports/YDSP%20Report_Evaluation%20of%20Artificial%20Intelligence%20Text%20Detection%20Models.pdf)  
14. \[2406.18259\] Detecting Machine-Generated Texts: Not Just "AI vs Humans" and Explainability is Complicated \- arXiv, accessed January 1, 2026, [https://arxiv.org/abs/2406.18259](https://arxiv.org/abs/2406.18259)  
15. SynthID: Tools for watermarking and detecting LLM-generated Text content\_copy, accessed January 1, 2026, [https://ai.google.dev/responsible/docs/safeguards/synthid](https://ai.google.dev/responsible/docs/safeguards/synthid)  
16. Watermarking AI-generated text and video with SynthID \- Google DeepMind, accessed January 1, 2026, [https://deepmind.google/blog/watermarking-ai-generated-text-and-video-with-synthid/](https://deepmind.google/blog/watermarking-ai-generated-text-and-video-with-synthid/)  
17. DetectRL: Benchmarking LLM-Generated Text Detection in Real-World Scenarios \- arXiv, accessed January 1, 2026, [https://arxiv.org/abs/2410.23746](https://arxiv.org/abs/2410.23746)  
18. Beyond Easy Wins: A Text Hardness-Aware Benchmark for LLM-generated Text Detection, accessed January 1, 2026, [https://arxiv.org/html/2507.15286v1](https://arxiv.org/html/2507.15286v1)  
19. Towards the Detection of Diffusion Model Deepfakes \- arXiv, accessed January 1, 2026, [https://arxiv.org/html/2210.14571v4](https://arxiv.org/html/2210.14571v4)  
20. Fourier Transform \- OpenCV Documentation, accessed January 1, 2026, [https://docs.opencv.org/4.x/de/dbc/tutorial\_py\_fourier\_transform.html](https://docs.opencv.org/4.x/de/dbc/tutorial_py_fourier_transform.html)  
21. \[2507.02995\] FreqCross: A Multi-Modal Frequency-Spatial Fusion Network for Robust Detection of Stable Diffusion 3.5 Generated Images \- arXiv, accessed January 1, 2026, [https://arxiv.org/abs/2507.02995](https://arxiv.org/abs/2507.02995)  
22. SFANet: Spatial-Frequency Attention Network for Deepfake Detection \- arXiv, accessed January 1, 2026, [https://arxiv.org/html/2510.04630v1](https://arxiv.org/html/2510.04630v1)  
23. DIRE for Diffusion-Generated Image Detection \- GitHub, accessed January 1, 2026, [https://github.com/vlgiitr/papers\_we\_read/blob/master/summaries/DIRE.md](https://github.com/vlgiitr/papers_we_read/blob/master/summaries/DIRE.md)  
24. ZhendongWang6/DIRE: \[ICCV 2023\] Official ... \- GitHub, accessed January 1, 2026, [https://github.com/ZhendongWang6/DIRE](https://github.com/ZhendongWang6/DIRE)  
25. FIRE: Robust Detection of Diffusion-Generated Images via Frequency-Guided Reconstruction Error \- arXiv, accessed January 1, 2026, [https://arxiv.org/html/2412.07140v3](https://arxiv.org/html/2412.07140v3)  
26. Noiseprint, a CNN-based camera model fingerprint \- grip-unina, accessed January 1, 2026, [https://grip-unina.github.io/noiseprint/](https://grip-unina.github.io/noiseprint/)  
27. grip-unina/noiseprint: Noiseprint, a CNN-based camera ... \- GitHub, accessed January 1, 2026, [https://github.com/grip-unina/noiseprint](https://github.com/grip-unina/noiseprint)  
28. The Stable Signature: Rooting Watermarks in Latent Diffusion Models \- Pierre Fernandez, accessed January 1, 2026, [https://pierrefdz.github.io/publications/stablesignature/](https://pierrefdz.github.io/publications/stablesignature/)  
29. \[2303.15435\] The Stable Signature: Rooting Watermarks in Latent Diffusion Models \- arXiv, accessed January 1, 2026, [https://arxiv.org/abs/2303.15435](https://arxiv.org/abs/2303.15435)  
30. invisible-watermark \- PyPI, accessed January 1, 2026, [https://pypi.org/project/invisible-watermark/](https://pypi.org/project/invisible-watermark/)  
31. Stability-AI/invisible-watermark-gpu \- GitHub, accessed January 1, 2026, [https://github.com/Stability-AI/invisible-watermark-gpu](https://github.com/Stability-AI/invisible-watermark-gpu)  
32. CIFAKE: Real and AI-Generated Synthetic Images \- Kaggle, accessed January 1, 2026, [https://www.kaggle.com/code/y5cy5c/cifake-real-and-ai-generated-synthetic-images](https://www.kaggle.com/code/y5cy5c/cifake-real-and-ai-generated-synthetic-images)  
33. CNN-generated images are surprisingly easy to spot... for now \- Sheng-Yu Wang, accessed January 1, 2026, [https://peterwang512.github.io/CNNDetection/](https://peterwang512.github.io/CNNDetection/)  
34. Leveraging Pre-Trained Visual Models for AI-Generated Video Detection \- arXiv, accessed January 1, 2026, [https://arxiv.org/html/2507.13224v1](https://arxiv.org/html/2507.13224v1)  
35. High-quality deepfakes have a heart\! \- Frontiers, accessed January 1, 2026, [https://www.frontiersin.org/journals/imaging/articles/10.3389/fimag.2025.1504551/full](https://www.frontiersin.org/journals/imaging/articles/10.3389/fimag.2025.1504551/full)  
36. A Comprehensive Review of Deepfake Detection Techniques Utilizing Remote Photoplethysmography \- IEEE Xplore, accessed January 1, 2026, [https://ieeexplore.ieee.org/iel8/6287639/10820123/11214405.pdf](https://ieeexplore.ieee.org/iel8/6287639/10820123/11214405.pdf)  
37. 29/12/2025 16:03 On Using rPPG Signals for DeepFake Detection: A Cautionary Note / D'Amelio, Alessandro \- IRIS Unimore, accessed January 1, 2026, [https://iris.unimore.it/retrieve/handle/11380/1314966/589046/ICIAP2023\_DeepFake\_rPPG.pdf](https://iris.unimore.it/retrieve/handle/11380/1314966/589046/ICIAP2023_DeepFake_rPPG.pdf)  
38. rPPG-Toolbox: Deep Remote PPG Toolbox (NeurIPS 2023\) \- GitHub, accessed January 1, 2026, [https://github.com/ubicomplab/rPPG-Toolbox](https://github.com/ubicomplab/rPPG-Toolbox)  
39. FaceForensics: A Large-scale Video Dataset for Forgery Detection in Human Faces \- Matthias Nießner, accessed January 1, 2026, [https://niessnerlab.org/papers/2018/z0faceforensics/faceforensics-large-scale.pdf](https://niessnerlab.org/papers/2018/z0faceforensics/faceforensics-large-scale.pdf)  
40. DeepfakeBench: A Comprehensive Benchmark of Deepfake Detection \- NIPS papers, accessed January 1, 2026, [https://proceedings.neurips.cc/paper\_files/paper/2023/file/0e735e4b4f07de483cbe250130992726-Paper-Datasets\_and\_Benchmarks.pdf](https://proceedings.neurips.cc/paper_files/paper/2023/file/0e735e4b4f07de483cbe250130992726-Paper-Datasets_and_Benchmarks.pdf)  
41. SCLBD/DeepfakeBench: A comprehensive benchmark of ... \- GitHub, accessed January 1, 2026, [https://github.com/SCLBD/DeepfakeBench](https://github.com/SCLBD/DeepfakeBench)  
42. Exposing Lip-syncing Deepfakes from Mouth Inconsistencies \- arXiv, accessed January 1, 2026, [https://arxiv.org/html/2401.10113v2](https://arxiv.org/html/2401.10113v2)  
43. DeepFakes AI — Improved Lip Sync Animations With Wav2Lip | by Chintan Trivedi \- Medium, accessed January 1, 2026, [https://medium.com/deepgamingai/deepfakes-ai-improved-lip-sync-animations-with-wav2lip-b5d4f590dcf](https://medium.com/deepgamingai/deepfakes-ai-improved-lip-sync-animations-with-wav2lip-b5d4f590dcf)  
44. FaceForensics++: Learning to Detect Manipulated Facial Images \- GitHub, accessed January 1, 2026, [https://github.com/ondyari/FaceForensics](https://github.com/ondyari/FaceForensics)  
45. Paper page \- The Deepfake Detection Challenge (DFDC) Preview Dataset \- Hugging Face, accessed January 1, 2026, [https://huggingface.co/papers/1910.08854](https://huggingface.co/papers/1910.08854)  
46. Official PyTorch implementation of "AASIST: Audio Anti-Spoofing using Integrated Spectro-Temporal Graph Attention Networks" \- GitHub, accessed January 1, 2026, [https://github.com/clovaai/aasist](https://github.com/clovaai/aasist)  
47. Uncovering the Real Voice: How to Detect and Verify Audio Deepfakes \- Medium, accessed January 1, 2026, [https://medium.com/htx-s-s-coe/uncovering-the-real-voice-how-to-detect-and-verify-audio-deepfakes-42e480d3f431](https://medium.com/htx-s-s-coe/uncovering-the-real-voice-how-to-detect-and-verify-audio-deepfakes-42e480d3f431)  
48. Hemgg/Deepfake-audio-detection \- Hugging Face, accessed January 1, 2026, [https://huggingface.co/Hemgg/Deepfake-audio-detection](https://huggingface.co/Hemgg/Deepfake-audio-detection)  
49. nii-yamagishilab/wav2vec-large-anti-deepfake \- Hugging Face, accessed January 1, 2026, [https://huggingface.co/nii-yamagishilab/wav2vec-large-anti-deepfake](https://huggingface.co/nii-yamagishilab/wav2vec-large-anti-deepfake)  
50. Towards Scalable AASIST: Refining Graph Attention for Speech Deepfake Detection \- arXiv, accessed January 1, 2026, [https://arxiv.org/html/2507.11777v1](https://arxiv.org/html/2507.11777v1)  
51. The official ElevenLabs MCP server \- GitHub, accessed January 1, 2026, [https://github.com/elevenlabs/elevenlabs-mcp](https://github.com/elevenlabs/elevenlabs-mcp)  
52. C2PA Implementation Guidance, accessed January 1, 2026, [https://spec.c2pa.org/specifications/specifications/2.2/guidance/Guidance.html](https://spec.c2pa.org/specifications/specifications/2.2/guidance/Guidance.html)  
53. contentauth/c2pa-python: Python binding for c2pa-rs library \- GitHub, accessed January 1, 2026, [https://github.com/contentauth/c2pa-python](https://github.com/contentauth/c2pa-python)
