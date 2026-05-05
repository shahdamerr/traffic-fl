# Review of Federated Learning and Traffic Prediction Papers

## 1-s2.0-S0045790624006050-main.pdf

### Summary
Road network traffic flow prediction: A personalized federated learning method based on client reputation Guowen Dai, Jinjun Tang *, Jie Zeng, Chen Hu, Chuyun Zhao Smart Transport Key Laboratory of Hunan Province, School of Transport and Transportation Engineering, Central South University, Changsha, 410075, China A R T I C L E I N F O Keywords: Traffic flow prediction Data privacy Federated learning Personalization A B S T R A C T Accurate traffic flow prediction can provide effective decision-making support for traffic man- agement, alleviate traffic congestion, and improve road traffic efficiency. Traffic flow data con- tains personal privacy information, such as vehicle trajectories, driving speed, etc. However, most existing research focuses on using all local data to jointly construct prediction models, facing data security and privacy issues. In response to these challenges, this paper presents a Personalized Federated Learning method based on Client Reputation for traffic flow ...

### Model Architecture & Pipeline
- With the increase in time steps, the personalized federated learning short-term traffic flow prediction model PFGCN-GRU exhibits superior performance and stability in terms of data adaptability, personalized training, privacy protection, and dynamic adjustments compared to the centrally trained GCN-GRU model.
- introduces pFedKT, an innovative personalized federated learning method that supports dual knowledge transfer—leveraging local supernets for historical local knowledge and contrastive learning for global knowledge—thereby enabling language models (LMs) to better balance personalization and generalization.
- Computers and Electrical Engineering 120 (2024) 109678 12 Table 1 shows the prediction performance of the personalized federated learning short-term traffic flow prediction model PFGCN- GRU based on client reputation, the local model GCN-GRU and other machine learning models on the Changsha LPR data set.
- 7 shows the comparison of the prediction performance between the PFGCN-GRU model trained under the personalized federated learning framework based on client reputation and the centralized training model GCN-GRU when the data volume of all clients is missing by 5 %, 10 %, 30 %, and 50 %.
- LSTM [4]: Long Short-Term Memory (LSTM) is a specialized type of recurrent neural network (RNN) that introduces gating mechanisms—including input, forget, and output gates—to regulate the flow of information, thereby effectively capturing long-term dependencies in time series data.

### Accuracies / Metrics Achieved
- Computers and Electrical Engineering 120 (2024) 109678 10 RMSE = ̅̅̅̅̅̅̅̅̅̅̅̅̅̅̅̅̅̅̅̅̅̅̅̅̅̅̅̅̅̅̅̅̅̅ 1 n ∑ n i=1 (̂Qi −Qi)2 √ (14) MAE = 1 n ∑ n i=1 |̂Qi −Qi| (15) Where n represents the number of samples in the dataset, Qi represents the actual value of the i th sample, ̂Qi represents the predicted value of the i th sample.
- This suggests that the personalized federated learning short-term traffic flow prediction model, PFGCN-GRU, based on client reputation, can achieve prediction accuracy that is on par with, or even surpasses, that of ordinary machine learning models in multi-step predictions.
- [32] propose a federated learning framework that incorporates layered protection and multiple aggregation methods to enhance data security while simultaneously addressing the challenges of accuracy, training time, and communication overhead.
- RMSE (Root Mean Square Error) and MAE (Mean Absolute Error) are frequently employed evaluation metrics in traffic flow prediction, serving to quantify the magnitude of errors between the predicted values of the model and the actual values.
- In addition, the CA-STIM module utilizes cross-attention to comprehensively capture the interaction of intelligent agents, integrating the spatial and temporal features of agent trajectories, thereby enhancing prediction accuracy [25].

### Baselines & Benchmarks
- With the increase in time steps, the personalized federated learning short-term traffic flow prediction model PFGCN-GRU exhibits superior performance and stability in terms of data adaptability, personalized training, privacy protection, and dynamic adjustments compared to the centrally trained GCN-GRU model.
- 5, with increasing time steps, the personalized federated learning short-term traffic flow prediction model PFGCN-GRU, based on client reputation, demonstrates superior and more stable prediction performance compared to the centrally trained GCN-GRU model.
- 6, the prediction performance of the centralized training machine learning model GCN-GRU selected in this sutdy is significantly better than other baseline models, and it has less abnormal data and a relatively even error distribution.
- FT-FedAvg involves simple fine-tuning on the basis of FedAvg, and while it improves the prediction performance of the global model across clients compared to ordinary federated learning models, the improvement is relatively modest.
- While the prediction perfor- mance of pFedme is superior to that of FT-FedAvg, its convergence speed is relatively slow, indicating that it necessitates a greater number of communication rounds to attain satisfactory performance.

---

## 1-s2.0-S2352864825000501-main.pdf

### Summary
Digital Communications and Networks 11 (2025) 724–733 Available online 7 May 2025 2352-8648/© 2025 Chongqing University of Posts and Telecommunications. Production and hosting by Elsevier B.V. on behalf of KeAi Communications Co. Ltd. This is an open access article under the CC BY-NC-ND license (http://creativecommons.org/licenses/by-nc-nd/4.0/). Contents lists available at ScienceDirect Digital Communications and Networks journal homepage: www.keaipublishing.com/dcan CMBA-FL: Communication-mitigated and blockchain-assisted federated learning for traﬃc flow predictions ✩ Kaiyin Zhu a, Mingming Lu a, ,∗, Haifeng Li b, Neal N. Xiong c, Wenyong He a a School of Computer Science and Engineering, Central South University, Changsha, 410083, China b School of Geosciences and Info-Physics, Central South University, Changsha, 410083, China c Department of Computer Science and Mathematics, Sul Ross State University, Alpine, TX 79830, USA A R T I C L E I N F O A B S T R A C T Keywords: Blockchain...

### Model Architecture & Pipeline
- Although more complex models such as BiGRU, BiLSTM [52], LSTM [53], and GNN generally outperform in terms of RMSE, it seems that the GRU-based Encoder-Decoder model is more eﬃcient in terms of training communication cost, indicating a better trade-off between prediction performance and communication overhead.
- Robustness under diﬀerent traﬃc conditions: The multi-layer struc­ ture of our Encoder-Decoder architecture enables our model to extract both short-term and long-term features eﬀectively, making it well-suited to handle sudden, short-term events such as traﬃc congestion or acci­ dents.
- Kim [44] proposes an FL method that is applied to the device based on the blockchain framework, stores the local gradient of each iteration in the block after verification and consensus, and analyzes the end-to­ end delay and the optimal block generation rate.
- [45] propose a blockchain-based privacy protection framework that eliminates the semi-honest assumptions of participants and employs encryption to protect data privacy, but it still carries the risk of a single­ point-of-failure of parameter servers.
- First, we use the Encoder-Decoder struc­ ture as the model architecture of each client, which eﬀectively improves the model’s ability to represent the temporal features in the traﬃc flow data and reduces the overall prediction error.

### Accuracies / Metrics Achieved
- Although more complex models such as BiGRU, BiLSTM [52], LSTM [53], and GNN generally outperform in terms of RMSE, it seems that the GRU-based Encoder-Decoder model is more eﬃcient in terms of training communication cost, indicating a better trade-off between prediction performance and communication overhead.
- Second, to reduce the communication overhead during the training stage, we introduce a pa­ rameter relevance detector, which avoids the additional communication overhead caused by irrelevant client parameter uploads and also ensures competitive accuracy.
- [30] propose a novel attention-based spatio temporal graph convolutional network method, that eﬀectively improves the TFP accuracy by modeling the three temporal attributes of traﬃc flow data separately.
- In terms of the extent to which model performance is af­ fected by changes in dataset scale, as the 𝛾increased, the RMSE of all CMBA-FL, CM-FedSeq2Seq, and CNFGNN all decreased, but to diﬀerent extents.
- RMSE = [ 1 𝑛 𝑛 ∑ 𝑖=1 (||𝑦𝑖−̂𝑦𝑖|| )2 ] 1 2 (9) MAE = 1 𝑛 𝑛 ∑ 𝑖=1 ||𝑦𝑖−̂𝑦𝑖|| (10) MAPE = 100% 𝑛 𝑛 ∑ 𝑖=1 |||| ̂𝑦𝑖−𝑦𝑖 𝑦𝑖 |||| (11) where ̂𝑦𝑖denotes the predicted value, and 𝑦𝑖denotes the true value.

### Baselines & Benchmarks
- Therefore, the proposed method CMBA­ FL does well to provide both higher accuracy and lower communication overhead compared to existing models.
- [17] use denoising algorithms to augment SVR prior to proposing a fuzzy C-mean neural network that eﬀectively improves the prediction accuracy.
- Although the RMSE of CMBA-FL is slightly worse compared to SAR, FNN, FC-LSTM, and STGCN, CMBA-FL is better each of them in both MAE and MAPE.
- [14] combine Principal Component Analysis (PCA) [15] and Sup­ port Vector Regression (SVR) [16] to predict traﬃc flows, and Tang et al.
- CM-FedSeq2Seq and CMBA-FL give the best results in several evaluation metrics compared to some typical-centralized meth­ ods.

---

## 1-s2.0-S2590005625000384-main.pdf

### Summary
Predicting traffic flow with federated learning and graph neural with asynchronous computations network Muhammad Yaqub a,*, Shahzad Ahmad b, Malik Abdul Manan b, Muhammad Salman Pathan c, Lan He a a School of Biomedical Sciences, Hunan University, Changsha, PR China b Faculty of Information Technology, Beijing University of Technology, PR China c School of Computing, Dublin City University, Ireland A R T I C L E I N F O Keywords: Traffic flow prediction Federated learning Graph convolutional network Intelligent transportation systems A B S T R A C T Real-time traffic flow prediction holds significant importance within the domain of Intelligent Transportation Systems (ITS). The task of achieving a balance between prediction precision and computational efficiency pre­ sents a significant challenge. In this article, we present a novel deep-learning method called Federated Learning and Asynchronous Graph Convolutional Network (FLAGCN). Our framework incorporates the principles of asynchron...

### Model Architecture & Pipeline
- The output layer utilizes the knowledge gained from the previous layers to generate accurate predictions for multiple time intervals (S)Given the complexity of ADGCN’s architecture and its extensive parameters, we suggest an optimized graph convolution operation for ADGCN.
- Connected spatial graphs We conducted an extensive series of experiments to assess the impact of adjusting the number of spatial graphs within the asynchronous spatial-temporal graph on the FLAGCN model’s performance, using both the METR-LA and PEMS08 datasets.
- The experimental results obtained from conducting tests on two distinct traffic datasets demonstrate that the utilization of FLAGCN leads to the optimization of both training and inference durations while maintaining a high level of prediction accuracy.
- By taking a thoughtful and strategic approach to partitioning the network and selecting model parameters, we were able to create a robust and effective FLAGCN model capable of accu­ rately predicting traffic flow patterns in diverse scenarios.
- The GraphFL algorithm, which forms the fundamental mech­ anism of the FLAGCN model, plays a crucial role in optimizing compu­ tational efficiency and minimizing communication overhead, all while maintaining high prediction accuracy.

### Accuracies / Metrics Achieved
- The experimental results obtained from conducting tests on two distinct traffic datasets demonstrate that the utilization of FLAGCN leads to the optimization of both training and inference durations while maintaining a high level of prediction accuracy.
- The GraphFL algorithm, which forms the fundamental mech­ anism of the FLAGCN model, plays a crucial role in optimizing compu­ tational efficiency and minimizing communication overhead, all while maintaining high prediction accuracy.
- We assessed its accu­ racy, robustness, and efficiency using metrics such as Root Mean Squared Error (RMSE), Mean Absolute Percentage Error (MAPE), Mean Absolute Error (MAE), R2 score , training time, and inference time.
- In comparison to the T-GCN model, the GMAN model has enhanced prediction accuracy because it makes use of the attention mechanism to precisely capture the influence of temporal and spatial elements on traffic flow.
- Notably, increasing the number of spatial graphs had the dual effect of improving prediction accuracy while reducing both training and Table 1 Comparison of prediction performance of different models.

### Baselines & Benchmarks
- The graph convolution operation requires the total parameter across all sub-models can be calculated as follows: Pall = ( m*N n )2 × n (1) Optimal segmentation of the network into sub-graphs (n) is guided by spatial distance constraints, where it’s noteworthy that n tends to be larger than m, signifying a greater number of spatial graphs compared to temporal ones.
- Tradi­ tional models, such as the historical average model (HA) [9] and the autoregressive integrated moving average model (ARIMA) [10], often make assumptions about data stability, making them less adept at managing non-linear traffic flow data.
- Table 1 displays the outcomes of the forecasting analysis, including the time for training and inference, for both the FLAGCN method and the baseline methods in the context of one-step ahead predictions with a 5- min interval.
- [15] created the Diffusion Convolutional Recurrent Neural Network (DCRNN), which simulates traffic flow as a diffusion pro­ cess on a directed graph and provides an effective technique to capture spatial information.
- In comparison to the T-GCN model, the GMAN model has enhanced prediction accuracy because it makes use of the attention mechanism to precisely capture the influence of temporal and spatial elements on traffic flow.

---

## Accelerating_Decentralized_Federated_Learning_With_Probabilistic_Communication_in_Heterogeneous_Edge_Computing.pdf

### Summary
Decentralized federated learning (DFL) has gained popularity for training machine learning models on massive data in edge computing, as it avoids the potential bottleneck of conven- tional parameter server architectures. However, the existing DFL solutions typically use deterministic topologies that struggle with both system heterogeneity and non-IID local data, resulting in high bandwidth costs and slow convergence rates. In this paper, we propose a novel mechanism called Communication-efﬁcient Decentralized Federated Learning (CEDFL) to accelerate model training. In CEDFL, each worker will communicate with each of its neighbors (i.e., model exchange) according to a certain probability at each epoch, so as to reduce bandwidth consump- tion. To this end, we then propose an efﬁcient algorithm to adaptively determine the optimal probability for each worker pair according to real-time system situations (e.g., data distri- bution and bandwidth resource). Our proposed mechanism has been ext...

### Model Architecture & Pipeline
- To simulate a decen- tralized federated learning edge computing system, we utilize 30 workers, with each implemented as a process in the system, along with a coordinator responsible for recording training performance and adjusting the probability of each link in the network.
- In order to grasp the essential fea- tures of CEDFL in comparison to previous algorithms, it is necessary to provide a brief overview of the frameworks of decentralized optimization and communication-efﬁcient dis- tributed optimization.
- CONCLUSION This paper introduces the CEDFL mechanism, which aims to tackle the challenges of decentralized federated learning (DFL) in edge computing, such as non-IID data, system heterogeneity, and limited bandwidth resources.
- The main contributions of this paper are summarized as follows: • We design a decentralized federated learning mechanism, named CEDFL, that is optimized for communication efﬁciency by utilizing probabilistic communication.
- The most related work to our paper is a decentralized federated learning approach, namely NetMax [12], that enables workers to communicate preferably through high-speed links to signiﬁcantly speed up the training process.

### Accuracies / Metrics Achieved
- Notably, when compared to benchmark methods, CEDFL has been observed to decrease com- pletion time by approximately 55% and enhance training accuracy by roughly 11% while operating under band- width constraints.
- Non-IID: Table IV presents the test accuracy performance of two models trained separately on IID and non-IID local data using BAT, Ring, NetMax, and CEDFL, with a ﬁxed number of model exchanges (e.
- As shown in Table III, given the target accuracy of 60%, the total number of model exchanges of CEDFL is 958, while that of BAT and Ring is 1,640 and 4,200, respectively, in the ﬁrst set of tests.
- : ACCELERATING DFL WITH PROBABILISTIC COMMUNICATION IN HETEROGENEOUS EDGE COMPUTING 9 • At the end of each round, we evaluate the global model on the test dataset and record the test accuracy.
- CEDFL has been shown to reduce completion time for model training by approximately 55% and improve test accuracy by 11% under the bandwidth constraint, compared to state-of-the-art solutions.

### Baselines & Benchmarks
- Notably, when compared to benchmark methods, CEDFL has been observed to decrease com- pletion time by approximately 55% and enhance training accuracy by roughly 11% while operating under band- width constraints.
- Performance Metrics and Benchmarks We use the following metrics to evaluate the performance of CEDFL and the baselines: • Training loss measures whether an FL algorithm can effectively achieve convergence.
- However, the framework mainly focuses on the asynchronous model training, which may lead to performance degradation compared to the synchronous scheme under the same number of training rounds [54].
- CEDFL has been shown to reduce completion time for model training by approximately 55% and improve test accuracy by 11% under the bandwidth constraint, compared to state-of-the-art solutions.
- 7–8, demonstrate that the bandwidth consumption and completion time of model training with CEDFL are signiﬁcantly lower than those of the other three benchmarks.

---

## Adaptive_Segmentation_Enhanced_Asynchronous_Federated_Learning_for_Sustainable_Intelligent_Transportation_Systems.pdf

### Summary
The proliferation of advanced embedded and communication technologies has facilitated the possibility of modern Intelligent Transportation System (ITS). The hierarchical nature of such large-scale and distributed systems brings obvious challenges in creating a scalable and sustainable computing environment, and hence the development and application of edge intelligence become critical. Federated learning (FL), as an emerging distributed machine learning paradigm, aims to offer secure knowledge sharing and effective learning across multiple devices. However, conventional FL may fall into trouble when facing large-scale and network-agnostic systems with fast moving devices and changing network attributes. In this study, we propose an Adaptive Segmentation enhanced Asynchronous Federated Learning (AS-AFL) model, aiming to improve the learning efficiency and reliability in sustainable ITS via a decentralized fashion. Specifically, a meta-learning based adaptive segmentation scheme is desig...

### Model Architecture & Pipeline
- On the other hand, the so-called inter- group asynchronous aggregation is more secure and robust across different groups because the framework has no single point of failure, which can effectively mitigate the network requirements as much as possible in the complex and resource- limited communication environment, especially in case of the connection or central node failure.
- i) A hybrid FL framework is constructed to enhance the efficiency and reliability in sustainable ITS, which includes a decentralized horizontal FL scheme among multiple vehicles via a peer-to-peer manner, and a ver- tical FL scheme that ensures secure knowledge/model sharing across different groups.
- Different to the traditional centralized FL, the proposed framework implements FL in a decentralized fashion to achieve edge intelligence in sustainable computing systems, where the model aggregation is performed in a peer-to-peer manner, without having a central server.
- [15] proposed an adaptive FL architecture with reinforcement learning for on-vehicle jamming attack detection, in which they used input from the FL model to update the Q-table, and employed the adaptive epsilon greedy policy in Q-learning to optimize the defense path.
- Experiment and evaluation results based on an open-source dataset demonstrate the outstanding learning and communication performance of our proposed model, compared with several conventional FL schemes in a distributed ITS application scenario.

### Accuracies / Metrics Achieved
- Evaluation results demonstrated that the proposed AS-AFL could outperform other conventional FL schemes in both learning accuracy and communication efficiency, within a simulated large-scale hierarchical vehicular network that consists of over 1000 nodes.
- Additionally, it is worth noticing that for both A-FedAvg and S-FedAvg, the accuracy and loss curves show more fluctuations, which means their learning processes are not very smooth.
- 03% accuracy and a great improvement in convergence time in comparison with Authorized licensed use limited to: Technische Universitaet Muenchen.
- The fully asynchronous approach achieves the best accuracy but also takes a long time for all the models to aggregate in an asynchronous fashion.
- Performances evaluated using the loss and accuracy curve for 200 iterations are illustrated in Fig.

### Baselines & Benchmarks
- The federated aggregation module of the proposed AS-AFL model, is evaluated against the state- of-the-art FL implementations with the purely synchronous update (later referred to as S-FedAvg) that represents the scenario of traditional centralized FL approach, and the purely asynchronous update (later referred to as A-FedAvg).
- : ADAPTIVE SEGMENTATION ENHANCED ASYNCHRONOUS FL FOR SUSTAINABLE ITS 6659 and sustainability issue for the targeted distributed and network-agnostic transportation system since existing practical vehicular networks do not have guaranteed stable connection or communication latency.
- Therefore, different gradient descent algorithms, including the synchronous algorithm SSGD, asynchronous algorithm ASGD, hybrid synchronous and asynchronous algorithm EASGD, and adaptive hybrid algorithm SGD-Gossip, are taken into account as the baseline methods.
- Experiment and evaluation results based on an open-source dataset demonstrate the outstanding learning and communication performance of our proposed model, compared with several conventional FL schemes in a distributed ITS application scenario.
- Meanwhile, the communication delay with LAN is also considered in the proposed method to reduce the convergence time, which is reflected in the fast convergence time compared with other GD methods.

---

## An_Active_Client_Selection_Scheme_Based_on_Blockchain_for_Federated_Learning_in_Shipping.pdf

### Summary
Federated Learning (FL) enables collaborative model training across maritime devices without the need to share raw data. However, challenges such as data heterogeneity and unreliable marine communications impede its performance and security. In this work, we propose a Blockchain-based Active Client Selection Strategy for FL in Shipping (BAFLS), which utilizes blockchain technology to create a secure and auditable environment for node registration and parameter exchange. A lightweight consensus algorithm is introduced to dynamically elect aggregation nodes based on residual energy, reputation, and computing power, improving fault tolerance and reducing resource consumption. Based on such, a Top-k active learning strategy is designed to select the most informative clients, balanc- ing data utility and privacy protection. Security evaluation and analysis demonstrate that BAFLS effectively resists aggregation attacks and privacy inference. Experimentations on FMNIST, HAR, and ShipNetwork10...

### Model Architecture & Pipeline
- Security Challenges of Federated Learning in Shipping In the field of shipping security, existing researches [34], [41] mainly depend on building network diagnosis and traffic monitoring models based on the centralized FL architecture, aiming to enhance the anti-attack ability of the shipping network while ensuring data privacy.
- Finally, in the above architecture, we design a client selection strategy based on Top-k AL, which can effectively identify and select the most valuable client nodes in each round of learning without revealing too much information, thus significantly improving the convergence speed of the global model.
- Building on the secure architecture, BAFLS integrates a Top-k AL- based client selection method to identify high-contribution clients while preserving local data privacy, mitigating data heterogeneity, improving convergence quality, and reducing communication overhead.
- For the classic FL framework, the communication process includes the following three stages: the aggregation server broadcasts global parameters to all clients, notifies part (ρ×m) of the selected clients to participate in the training, and the Fig.
- We first introduce the system architecture and node registration process in IV-A, followed by a consensus-driven aggregation node election mechanism in IV-B, and finally describe the client selection and model aggregation process in IV-C and IV-D.

### Accuracies / Metrics Achieved
- (18) To this end, the communication efficiency of the two schemes can be expressed as follows: ( COFL × RoundsT Acc FL COBAFLS × RoundsT Acc BAFLS, (19) RoundsT Acc FL and RoundsT Acc BAFLS represent the number of global training rounds in which the global model accuracy of FedAvg and BAFLS reached T Acc for the first time, respectively.
- The results indicate that BAFLS-LC, BAFLS-MS, and BAFLS-Entropy outperform baselines in both accuracy and convergence efficiency, with advantages increasing as label distribution non-uniformity rises (dir(1.
- Evaluation of Communication Efficiency In the field of shipping, energy resources are incredibly scarce, so the trade-off between the accuracy of the model and the overhead of communication is essential.
- 4% higher accuracy, reduces convergence rounds by up to 44%, and consistently lowers communication overhead compared to the baseline under various degrees of label and feature heterogeneity.
- Learning from this type of sample helps the model more quickly recognize the characteristics of different categories, improving its accuracy and generalization ability.

### Baselines & Benchmarks
- Due to the strong sensitivity of the hash function to the input, at the current time Timestampt′, the probability that a replay attack will succeed by replaying a record from a historical Timestampt is: Pr′ 2[A ∈St] = Pr[Hash(P I DA ∥nonce ∥Timestampt) = Hash(P I DA ∥nonce ∥Timestampt′)] ≈1 2κ ≤negl1(κ), (16) where κ is the output length of the hash function, and for SHA- 256, κ = 256.
- (18) To this end, the communication efficiency of the two schemes can be expressed as follows: ( COFL × RoundsT Acc FL COBAFLS × RoundsT Acc BAFLS, (19) RoundsT Acc FL and RoundsT Acc BAFLS represent the number of global training rounds in which the global model accuracy of FedAvg and BAFLS reached T Acc for the first time, respectively.
- TABLE II TEST ACCURACIES(%) ± STDS AND THE NUMBER OF TRAINING ROUNDS FOR THE FIRST TIME TO ACHIEVE THE SPECIFIED ACCURACIES FOR DIFFERENT SCHEMES UNDER FMNIST WITH DIFFERENT DEGREES OF HETEROGENEITY Table III shows the HAR results with feature skew, which reports the classification results at participation rates of 0.
- This is because the increased imbalance in the label distribution causes the client selection strategies of existing methods to fail: the random selection of FedAvg fails to capture key samples, while the full-sample average metric of ACFL favors small sample clients, exacerbating local overfitting.
- : ACTIVE CLIENT SELECTION SCHEME BASED ON BLOCKCHAIN FOR FL IN SHIPPING 20681 TABLE V TEST ACCURACIES (%) ± STDS AND THE NUMBER OF TRAINING ROUNDS FOR THE FIRST TIME TO ACHIEVE THE SPECIFIED ACCURACIES FOR DIFFERENT SCHEMES UNDER FMNIST WITH DIFFERENT NETWORK SIZES Fig.

---

## applsci-13-05270-v2.pdf

### Summary
This paper proposes utilizing federated learning (FL), a distributed learning paradigm, to process large, decentralized, and heterogeneous edge data in the context of Internet of Things (IoT) devices. However, heterogeneity and high communication costs are two primary challenges that hinder the efﬁcacy of federated learning. To overcome these challenges, we have designed an algorithm, FedACADMM, which applies the adaptive consensus alternating direction method of multipliers (ACADMM) to federated learning clients (i.e., the edge mobile devices) to tackle the heterogeneity problem in federated networks. Importantly, the cost per round of communication for FedACADMM remains consistent with FedAvg and FedProx without adding any extra workload. Furthermore, our experimental results demonstrate that FedACADMM outperforms baseline meth- ods with a realistic set of federated datasets, displaying enhanced convergence robustness. Notably, in highly heterogeneous scenarios, FedACADMM exhibits si...

### Model Architecture & Pipeline
- Hence, in this article, we decompose the objective function of federated learning into sub-problems using this method, min x∈Rd ( F(x) := f (x) + g(x) = 1 n m ∑ i=1 fi(x) + g(x) ) (2) where m is the number of devices and the weighted local training loss is represented by fi, which is considered to be L-smooth and nonconvex, and g is a proper, closed, and convex regularizer.
- Because convergence speed and ﬁnal performance are important indicators of federated learning, we computed the performance attained at a speciﬁed number of rounds and the number of communication rounds required for the algorithm to reach the test target accuracy.
- The data they generate are also diverse, and they are widely distributed across the Internet of Things; although they want to participate in federated learning training, they do not want their privacy to be compromised.
- Datasets Task # Parameter Model MNIST Handwritten recognition 1,663,370 2-layer CNN + FC CIFAR-10 Image classiﬁcation 1,105,098 2-layer CNN + FC FMNIST Image classiﬁcation 1,663,370 2-layer CNN + FC Data Distribution.
- The ultimate aim of federated learning is typically to minimize the following objective function: min w F(w), where F(w) := m ∑ k=1 pkFk(w) (1) Here, m represents the total number of devices, pk ≥0, and ∑k pk = 1.

### Accuracies / Metrics Achieved
- In Figure 5, the corresponding train loss of test accuracy in Figure 4 is shown, and it can be seen that the convergence performance of FedACADMM is equal to the other two algorithms in most heterogeneous cases, and the convergence speed can be better than the other two algorithms in homogeneous cases.
- Because convergence speed and ﬁnal performance are important indicators of federated learning, we computed the performance attained at a speciﬁed number of rounds and the number of communication rounds required for the algorithm to reach the test target accuracy.
- When the number of clients is small, the test accuracy of FedACADMM is almost the same as that of the other two algorithms, but its advantage becomes obvious as the number of clients increases; as such, we guess that it is suitable for large-scale communication.
- Under the homogeneous data distribution method, al- most all experimental methods can quickly achieve the test target accuracy, and the increase in the number of clients enables the advantages of our proposed method to be reﬂected.
- In Table 4, we report the number of communication rounds required by each of the three algorithms to attain the target accuracy for three distinct datasets, varying data distributions, and a range of client population sizes.

### Baselines & Benchmarks
- • By comparing with baselines in multiple popular datasets, we demonstrate that our federated optimization algorithm improves communication efﬁciency and is robust to client heterogeneity, especially in highly heterogeneous situations.
- A communication-efﬁcient method, q-FFL, is inspired by techniques of fair resource allocation in wireless networks, and it alters the objective function of FedAvg by modifying the weights of different devices [25].
- Across all range of values tried, we observe that the proposed method achieves test target accuracy almost faster than baselines in both homogeneous and heterogeneous experimental settings.
- Furthermore, our experimental results demonstrate that FedACADMM outperforms baseline meth- ods with a realistic set of federated datasets, displaying enhanced convergence robustness.
- Our experiments involved handwriting recognition and image classiﬁcation using three well-known benchmark datasets: MNIST [35], Fashion MNIST (FMNIST) [36], and CIFAR-10 [37].

---

## FedAGCN.pdf

### Summary
Applied Soft Computing 138 (2023) 110175 Contents lists available at ScienceDirect Applied Soft Computing journal homepage: www.elsevier.com/locate/asoc FedAGCN: A traffic flow prediction framework based on federated learning and Asynchronous Graph Convolutional Network Tao Qi a, Lingqiang Chen a, Guanghui Li a,∗, Yijing Li a, Chenshu Wang b a School of Artificial Intelligence and Computer Science, Jiangnan University, Wuxi, Jiangsu 214122, China b School of Computing Science and Communication Engineering, Jiangsu University, Zhenjiang, Jiangsu, 212000, China a r t i c l e i n f o Article history: Received 19 April 2022 Received in revised form 26 January 2023 Accepted 3 March 2023 Available online 8 March 2023 Keywords: Traffic flow prediction Graph convolutional network Asynchronous spatial–temporal correlation Federated learning a b s t r a c t Accurate and real-time traffic flow prediction is an essential component of the Intelligent Transporta- tion System (ITS). Balancing the pre...

### Model Architecture & Pipeline
- Improved asynchronous spatial–temporal graph convolutional network based on federated learning From the perspective of information dissemination, the tem- poral dependence of the same traffic node at different times can be regarded as the information dissemination process on the graph.
- In the original ADGCN model, combining multiple spatial graphs into asynchronous spatial–temporal graphs will lead to a rapid expansion of the network scale and a sharp increase in the number of parameters required by the model, making the model over-fitting and difficult to train.
- It can be seen from Table 3 that the prediction accuracy of FedAGCN-all is much lower than that of the FedAGCN model, which shows that the participation of GCN related parameters in the global parameter update process has a negative impact on the prediction accuracy of the model.
- Recent studies mostly use the graph neural network (GNN) to model the spatial relationship of traffic data and use the recurrent neural network (RNN) or convolutional neural network (CNN) to model the temporal rela- tionship of traffic data [6–8].
- Graph Wavenet uses one-dimensional dilated causal convolution instead of RNN to model the time dependence of traffic data, which better reduces the calculation time of the model, but still cannot meet the needs of real-time prediction tasks.

### Accuracies / Metrics Achieved
- It can be seen from Table 3 that the prediction accuracy of FedAGCN-all is much lower than that of the FedAGCN model, which shows that the participation of GCN related parameters in the global parameter update process has a negative impact on the prediction accuracy of the model.
- Experiments were conducted on two public traffic datasets, and the results showed that FedAGCN effectively reduced the training and inference time of the model while maintaining considerable prediction accuracy.
- In this study, we divide the traffic network into K sub-graphs and propose FedAGCN model, which minimizes the model’s deployment cost and time consumption while ensuring the prediction accuracy of the model.
- However, the traditional federated learning algorithms cannot make use of the spatial structure of the traffic network, and its prediction accuracy is poor compared with the existing deep learning model.
- The GraphFed algorithm is the core mechanism of the FedAGCN model, which can effectively reduce the calculation time of the model and the communication overhead while ensuring high prediction accuracy.

### Baselines & Benchmarks
- Some researchers have used linear regression models in the field of traffic prediction, such as the his- torical average model (HA) [10] and the autoregressive integrated moving average model (ARIMA) [11].
- However, the traditional federated learning algorithms cannot make use of the spatial structure of the traffic network, and its prediction accuracy is poor compared with the existing deep learning model.
- [17] proposed a Time Graph Convolutional Network (T-GCN) model, representing the traffic network as a graph structure, and com- bined GCN and GRU to model the spatial–temporal dependence of traffic data.
- Experimental results Table 2 records one-step ahead (in a 5 min interval) forecast- ing results, training and inference time for FedAGCN and baseline methods.
- [18] first modeled traffic flow as a diffusion process on a directed graph and proposed a Diffusion Convo- lutional Recurrent Neural Network (DCRNN) model.

---

## Federated_Learning_for_Intelligent_Transportation_Systems_Use_Cases_Open_Challenges_and_Opportunities.pdf

### Summary
IEEE INTELLIGENT TRANSPORTATION SYSTEMS MAGAZINE • 18 • MAY/JUNE 2025 1939-1390/24©2024IEEE Digital Object Identifier 10.1109/MITS.2024.3451479 Date of publication 16 September 2024; date of current version 8 May 2025. *Corresponding author Abstract—Intelligent transportation systems (ITSs) leverage a network of interconnected infrastructures utilizing advanced technologies to improve traffic management and safety. Federated learning (FL) has emerged as a pivotal method within ITSs, enabling decentralized collaborative model training without direct data sharing, thus preserving privacy and enhancing system efficiency. This article explores the integration of FL in ITSs, focusing on FL’s application in traffic flow prediction, trajectory prediction, park- ing space estimation, and traffic target recognition. Despite its potential, FL deployment faces challenges, including data heterogeneity, communication and bandwidth constraints, and resource limitations on edge devices. Addressing th...

### Model Architecture & Pipeline
- These models often harness the capabilities of deep neural networks, particularly recurrent neural networks (RNNs) and trans- formers, which have shown effectiveness in forecasting trajectories of various entities, such as vehicles and pe- destrians, and in capturing their behavioral patterns [33].
- ■ Standardization and benchmarking: Developing standardized benchmarks, evaluation metrics, and comprehensive evaluation frameworks specifically for FL in ITSs would facilitate comparisons among different approaches and accelerate progress in ad- dressing all identified challenges.
- FedSTN’s architecture consists of modules for capturing long-term spatial–temporal dynamics, sharing short-term information while preserving privacy through homomorphic encryption and recognizing semantic fea- tures like non-Euclidean connections and points of inter- est.
- ■ Adaptive resource allocation: This relates to designing adaptive FL frameworks that can dynamically adjust to varying network conditions and bandwidth availability, intelligently allocating tasks and prioritizing critical operations across the network.
- The proposed method, hierarchical trajectory planning method with deep reinforcement learning in the federated learning scheme (HALEOS) integrates deep RL (DRL) with optimization- based techniques to generate efficient and accurate parking trajectories.

### Accuracies / Metrics Achieved
- This approach not only addresses privacy concerns but also has the potential to mitigate data quality issues by leveraging diverse data sources and local expertise, ultimately en- hancing the accuracy and reliability of traffic flow predic- tions in ITSs.
- Their study integrates factors such as holidays, weekends, and weekdays to enhance the dataset, with results showing that the LeNet-5 architecture performs best in a traditional setup, while the LSTM model achieves the highest accuracy in the FL setup.
- The framework has been tested on two public traffic datasets, and the results demonstrated that FedAGCN not only maintains high prediction accuracy but also signifi- cantly reduces both the training and inference times of the DL model.
- Extensive case studies show that FedGRU’s prediction accuracy ex- ceeds 90%, confirming its efficacy in providing accurate and timely traffic predictions without compromising the privacy and security of raw data.
- Experiments demonstrate that Automatic Trajectory Prediction Model Design under Federated Learning Framework (ATPFL) outperforms tra- ditional models by providing higher accuracy and efficien- cy in predictions.

### Baselines & Benchmarks
- ■ Standardization and benchmarking: Developing standardized benchmarks, evaluation metrics, and comprehensive evaluation frameworks specifically for FL in ITSs would facilitate comparisons among different approaches and accelerate progress in ad- dressing all identified challenges.
- The proposed method, hierarchical trajectory planning method with deep reinforcement learning in the federated learning scheme (HALEOS) integrates deep RL (DRL) with optimization- based techniques to generate efficient and accurate parking trajectories.
- Their evaluations with data from mul- tiple parking lots demonstrate significant improvements in model variance reduction, training speed, and forecasting performance compared to baseline methods.
- Their secure FL system groups users into clusters and uses two aggregation methods to form global and cus- tomized local models, showing impressive performance on established NCD benchmarks.
- The proposed GCN outperforms several benchmark models, including LSTM, convolutional neural network (CNN), Chebyshev spectral CNN, and graph attention net- work models.

---

## Federated_Meta-Learning_on_Graph_for_Traffic_Flow_Prediction.pdf

### Summary
Trafﬁc ﬂow is considered as a critical feature of in- telligent transportation systems (ITS). Accurately forecasting fu- ture vehicular volumes is an effective means of mitigating traf- ﬁc congestion. However, the nonlinear and complex trafﬁc ﬂow characteristics make the traditional approaches unable to achieve satisfactory prediction performance. Although existing methods based on deep learning models have improved the accuracy of trafﬁc ﬂow prediction, the spatio-temporal features of trafﬁc ﬂow data are still not fully explored. Moreover, existing methods pay little attention to the task of training models in a decentralized environment where data are distributed across multiple clients. To solve the problems mentioned above, we propose a novel network model called Graph Transformer Attention Network (GTAN) for trafﬁc ﬂow prediction, which can effectively extract trafﬁc ﬂow’s temporal and spatial characteristics by considering all node lo- cations’ information in the trafﬁc networks....

### Model Architecture & Pipeline
- However, the general federated learning process extracts part of the client to participate in training every round to save training costs, at the same time, during the meta-training phase, only part of tasks is selected for the inner loop, this can lead to clients who are not selected in a given round being unable to update their local private encoding and decoding states.
- Unlike the above method, our GTAN model achieves better trafﬁc ﬂow prediction performance than all the above methods because the Graph-Transformer module can comprehensively consider the information of all moments of the trafﬁc ﬂow sequence and synchronously extract their deep spatio-temporal TABLE V THE COMPUTATION COST ON THE PEMSD4 DATASET features.
- For example, a Diffusion Convolutional Recurrent Neural Network (DCRNN) was proposed in [1] to capture the spatial and temporal correlations of trafﬁc data and a Spatial-Temporal Graph Convolutional Network (ST-GCN) was presented in [2] that combines graph convolution and time convolution to capture the spatial-temporal correlation of trafﬁc ﬂow.
- Then, we propose a training strategy called Graph Federated Meta-learning (FedGM), solving the problem of topological heterogeneity by combining meta-learning and federated learning, to achieve an optimal initial- ization model which can quickly adapt to different trafﬁc networks under low communication cost.
- To solve the problems mentioned above, we propose a novel network model called Graph Transformer Attention Network (GTAN) for trafﬁc ﬂow prediction, which can effectively extract trafﬁc ﬂow’s temporal and spatial characteristics by considering all node lo- cations’ information in the trafﬁc networks.

### Accuracies / Metrics Achieved
- Mean Absolute Error (MAE): It is the mean of the absolute differences between the predicted and actual values, calculated by taking the absolute difference of predicted and actual values and averaging over the samples, as MAE = 1 n n  i=1 |ˆyi −yi|, (11) where ˆy represents the predicted value, n represents the number of samples, and y represents the actual value.
- For FedGM-RI, the encoder-decoder is randomly initialized in each round of training, allowing the other parts of the model to adapt to the encoder-decoder’s randomly initialized parameters, giving it some adaptability for non-federated clients, while the average MAE on federated clients is 52.
- Experimental results demonstrate that compared with advanced trafﬁc ﬂow prediction methods, GTAN achieves higher prediction accuracy, while FedGM can train a meta-learning model that quickly adapts to different graph topologies no matter the client participates the model training or not.
- 9, it can be observed that FedGM achieved the highest accuracy after just two rounds of training, as shown in the zoomed-in plot, indicating that the meta-learning model can quickly adapt to data with different topological structures through the meta-training process.
- proposed the STSGCN, which pre-stacked multiple adjacency matricesandusedgraphconvolutionoperationtoaggregatenode information from multiple time segments to extract complete spatial-temporal correlation information and improve model prediction accuracy [22].

### Baselines & Benchmarks
- For example, a Diffusion Convolutional Recurrent Neural Network (DCRNN) was proposed in [1] to capture the spatial and temporal correlations of trafﬁc data and a Spatial-Temporal Graph Convolutional Network (ST-GCN) was presented in [2] that combines graph convolution and time convolution to capture the spatial-temporal correlation of trafﬁc ﬂow.
- : FEDERATED META-LEARNING ON GRAPH FOR TRAFFIC FLOW PREDICTION 19533 TABLE I THE NUMBER OF SENSOR NODES HELD BY CLIENTS PARTICIPATING IN META-LEARNING TABLE II THE NUMBER OF SENSOR NODES HELD BY CLIENTS THAT ARE NOT PARTICIPATING IN META-LEARNING BUT PARTICIPATING IN META-TESTING data are used as the model input and output.
- In the early days, researchers used methods based on traditional time series analysis models, such as History Average Model (HA) [4], Autoregressive Integrated Moving Average Model (ARIMA) [5], Vector Auto Regression Model (VAR) [6], and their related extended models, which were widely used time series methods.
- Although machine learning methods have demon- strated their advantages in processing complex data and achiev- ing better prediction results compared to traditional methods, the complex spatial and temporal characteristics of trafﬁc ﬂow data still signiﬁcantly impact the performance of machine learning methods.
- Although deep learning-based methods have made great progress compared to previous methods, separately extracting time and spatial correlations ignores the complex dynamic characteristics of spatio-temporal relationships, and some mod- els have slow computation speed and high application costs.

---

## FedSTDN_A_Federated_Learning-Enabled_Spatial-Temporal_Prediction_Model_for_Wireless_Traffic_Prediction.pdf

### Summary
Wireless Trafﬁc Prediction (WTP) plays a signiﬁcant role in achieving intelligent resource management forcommuni- cation systems. However, WTP still faces challenges such as in- accurate prediction resulting from the complex spatial-temporal characteristics due to user mobility, high communication overhead caused by the complexity of the prediction model, and user pri- vacy issues stemming from Centralized Learning (CL). To address the aforementioned issues, this paper proposes a WTP frame- work under the Federated Learning (FL) strategy called Feder- ated Spatial-Temporal Dual-attention based Network (FedSTDN). Aiming at improving communication efﬁciency and simultaneously representing various wireless trafﬁc patterns, a data augmentation- based clustering algorithm is adopted, which groups cells into different regions using a small augmented dataset, facilitating subsequent processing. To improve prediction performance, a local prediction model based on Convolutional Neural Network (...

### Model Architecture & Pipeline
- 2) ComparedwithtraditionalFedAvgandFedAttalgorithms, FedSTDN introduces a clustering strategy to make the patterns of wireless trafﬁc more speciﬁc, the local model can characterize short-term and long-term dependencies, and the dual attention mechanism greatly reduces the heterogeneity of data.
- [20] developed an attention- based ConvLSTM module to extract the spatial and short-term temporal features, where the attention mechanism is properly designed to adaptively allocate different levels of attention to a trafﬁc ﬂow sequence at different times.
- Subsequently, in order to further en- hancethepredictionperformance,theresearchersdesignvarious network architectures to represent temporal and spatial depen- dencies separately, and use effective tools such as attention mechanisms to improve accuracy.
- Therefore, to cope with these two issues, this paper pro- poses a WTP framework called the Federated Spatial-Temporal Dual-attention Network (FedSTDN), through which multiple base stations (BSs) collaboratively train a high-quality predic- tion model.
- [16] adopted a densely con- nected CNN to capture the spatial-temporal dependencies of cell trafﬁc by treating trafﬁc data as frame-by-frame images, and proposed a parameter matrix-based fusion scheme to capture spatial-temporal dependencies.

### Accuracies / Metrics Achieved
- Subsequently, in order to further en- hancethepredictionperformance,theresearchersdesignvarious network architectures to represent temporal and spatial depen- dencies separately, and use effective tools such as attention mechanisms to improve accuracy.
- As shown in the ﬁgure, the trends of MSE and MAE for Call, SMS, and Internet trafﬁc with respect to grid size are basically consistent across different datasets, generally showing a trend of ﬁrst decreasing and then slowly increasing.
- For the Trentino dataset, introducing the clustering strategy also results in better prediction performance compared to not intro- ducing it, especially for Internet trafﬁc, where the normalized MSE is improved by up to 4.
- [32] proposed a novel joint MEC server selection and dataset management mechanism for FL-based mobile trafﬁc prediction over MEC servers, including an optimization prob- lem for balancing the accuracy-cost tradeoff.
- Due to the introduction of 1D CNN and KAN, the proposed FedSTDN may exhibit higher complexity compared to other methods, although it outperforms them in terms of both accuracy and communication efﬁciency.

### Baselines & Benchmarks
- The reasons for the superiority of FedSTDN over other baseline methods are as follows: 1) Compared to the STDRN method based on the CL strat- egy, which may be affected by uneven data distribution and computational bottlenecks, the FL strategy effectively leverages distributed data and computing resources across devices, adapting to the characteristics of different devices and regions.
- 2) ComparedwithtraditionalFedAvgandFedAttalgorithms, FedSTDN introduces a clustering strategy to make the patterns of wireless trafﬁc more speciﬁc, the local model can characterize short-term and long-term dependencies, and the dual attention mechanism greatly reduces the heterogeneity of data.
- For the Trentino dataset, introducing the clustering strategy also results in better prediction performance compared to not intro- ducing it, especially for Internet trafﬁc, where the normalized MSE is improved by up to 4.
- Due to the introduction of 1D CNN and KAN, the proposed FedSTDN may exhibit higher complexity compared to other methods, although it outperforms them in terms of both accuracy and communication efﬁciency.
- As can be seen from Table II, our proposed method FedSTDN outperforms all baseline methods in all types of wireless trafﬁc and across both datasets, even when only 1% of augmented data is shared.

---

## futureinternet-17-00109-v2.pdf

### Summary
This paper presents the Mobility-Aware Client Selection (MACS) strategy, de- veloped to address the challenges associated with client mobility in Federated Learning (FL). FL enables decentralized machine learning by allowing collaborative model train- ing without sharing raw data, preserving privacy. However, client mobility and limited resources in IoT environments pose significant challenges to the efficiency and reliability of FL. MACS is designed to maximize client participation while ensuring timely updates under computational and communication constraints. The proposed approach incorporates a Mobility Prediction Model to forecast client connectivity and resource availability and a Resource-Aware Client Evaluation mechanism to assess eligibility based on predicted latencies. MACS optimizes client selection, improves convergence rates, and enhances overall system performance by employing these predictive capabilities and a dynamic resource allocation strategy. The evaluation includ...

### Model Architecture & Pipeline
- These mechanisms enable MACS to operate efficiently in IoT networks with high mobility and diverse resource constraints, improving convergence speed, reducing communication overhead, and enhancing the robustness of federated learning in dynamic environments.
- This work proposes that the Mobility- Aware Client Selection (MACS) framework tackles these issues by integrating mobility Future Internet 2025, 17, 109 5 of 19 prediction, resource-aware client evaluation, and dynamic selection mechanisms.
- While there are challenges related to computational overhead and high-mobility scenarios, potential optimizations and enhancements promise to make MACS an even more effective tool for client selection in federated learning applications.
- Future work will explore integrating deep learning models, including Recurrent Neural Networks (RNNs), Long Short-Term Memory (LSTMs), and Transformers, to refine mobility prediction by analyzing sequential patterns.
- System Models and Problem Formulation To address the challenges posed by client mobility and resource variability in federated learning (FL), this paper proposes the Mobility-Aware Client Selection (MACS) strategy.

### Accuracies / Metrics Achieved
- Accuracy Evaluation on CIFAR and MNIST Datasets The Mobility-Aware Client Selection (MACS) strategy was evaluated against Static client selection, Random selection, Reinforcement Learning-based FL (RL-based), and Deep Learning-based FL (DL-based) using the CIFAR and MNIST datasets.
- v log2(1/η) indicates the number of iterations needed to achieve the desired accuracy η, where v is defined as follows: v = 2 (2 −Lδ)δγ, (2) with δ being the learning rate, and L and γ depending on the eigenvalues of the Hessian matrix of the loss function.
- For instance, integrating advanced predictive models like Kalman filters for state estimation or LSTM networks to analyze sequential mobility patterns could improve selection accuracy in dynamic IoT environments.
- The results confirm that MACS improves accuracy and stability in federated learning by dynamically selecting clients based on mobility and resource availability, making it effective in dynamic IoT environments.
- MACS achieved the highest final accuracy, reaching approximately 95%, outperforming RL-based (92%), DL-based (91%), Static selection (85%), and Random selection (80%).

### Baselines & Benchmarks
- The evaluation includes comparisons with advanced baselines such as Reinforcement Learning-based FL (RL-based) and Deep Learning-based FL (DL- based), in addition to Static and Random selection methods.
- Traditional methods like Federated Averaging (FedAvg) [6] assume that clients stay put and do not change much, which does not work in IoT networks where everything is constantly moving.
- While MACS introduces additional computation for mobility prediction and resource-aware selection, it remains lightweight compared to Deep Reinforcement Learning-based approaches.
- For instance, an IoT device with limited computational resources will take longer to process updates compared to a more powerful device.
- Section 5 presents simulation results, comparing MACS with baseline methods.

---

## futureinternet-17-00315.pdf

### Summary
Federated learning (FL) has emerged as a powerful approach for privacy-preserving model training in autonomous vehicle networks, where real-world deployments rely on multiple roadside units (RSUs) serving heterogeneous clients with intermittent connectivity. While most research focuses on single-server or hierarchical cloud-based FL, multi-server FL can alleviate the communication bottlenecks of traditional setups. To this end, we propose an edge-based, multi-server FL (MS-FL) framework that combines performance-driven aggregation at each server—including statistical weighting of peer updates and outlier mitigation—with an application layer handover protocol that preserves model updates when vehicles move between RSU coverage areas. We evaluate MS-FL on both MNIST and GTSRB benchmarks under shard- and Dirichlet-based non-IID splits, comparing it against single-server FL and a two-layer edge-plus-cloud baseline. Over multiple communication rounds, MS-FL with the Statistical Performance-...

### Model Architecture & Pipeline
- By leveraging inter-RSU collab- oration, our framework accelerates convergence and enhances robustness in highly mobile vehicular networks; • We develop and evaluate server-level, performance-based aggregation strategies whereby each FL server first assesses incoming peer models’ accuracy and loss against its own validation data and then selectively incorporates these updates.
- To this end, we propose an edge-based, multi-server FL (MS-FL) framework that combines performance-driven aggregation at each server—including statistical weighting of peer updates and outlier mitigation—with an application layer handover protocol that preserves model updates when vehicles move between RSU coverage areas.
- While the server plays a critical role in orches- trating the learning process by aggregating local model updates and ensuring data privacy and system security, the potential of multi-server architectures—particularly when explicitly considering server performance metrics such as accuracy and loss—remains underexplored.
- MS-FL In our MS-FL framework, the total transmission latency in each communication round denoted by τMS-FL(t) comprises three key components: the local uploading time from clients to their regional FL servers, the downloading time from regional servers to clients, and the inter-server (RSU-to-RSU) transmission time.
- Then, the vehicle compares these require- ments against its local model metadata; if they match, it packages its latest parameters and metadata and uploads them to the new server, which enqueues them into the aggregation pipeline so that the vehicle can resume participation or receive further training tasks.

### Accuracies / Metrics Achieved
- Our goal is to enable this federation of servers and vehicles to collaboratively train a shared ML/DL model with high accuracy, while ensuring the following: • Ensuring robust aggregation, despite heterogeneous updates; • Respecting vehicles’ mobility, which causes frequent handovers and variable participation; • Minimizing communication overhead, given limited edge-network bandwidth.
- By leveraging inter-RSU collab- oration, our framework accelerates convergence and enhances robustness in highly mobile vehicular networks; • We develop and evaluate server-level, performance-based aggregation strategies whereby each FL server first assesses incoming peer models’ accuracy and loss against its own validation data and then selectively incorporates these updates.
- While the server plays a critical role in orches- trating the learning process by aggregating local model updates and ensuring data privacy and system security, the potential of multi-server architectures—particularly when explicitly considering server performance metrics such as accuracy and loss—remains underexplored.
- Then the accuracy of the model w(s) against D(s) v is computed as follows: As = 1 J J ∑ j=1 I(yj= argmax(ws(xj))), (1) where ws(xj) is the model’s output for input xj, and I(·) is the indicator function that returns 1 if the predicted label (obtained via argmax) matches the true label yj, and 0 otherwise.
- Although SPAA, DWAA, and SA exhibit slightly higher runtimes than WA due to additional evaluation and aggregation steps, the substantial gains in accuracy and recall justify the increased computational cost, making SPAA a highly effective method for MS-FL in environments with data heterogeneity.

### Baselines & Benchmarks
- While both approaches exploit overlapping areas, MS-FedAvg distinguishes itself by providing a thorough convergence analysis under non-convex settings and by incorporating algorithmic refinements—such as biased client sampling—to enhance both theoretical guarantees and empirical performance in heterogeneous network environments.
- Comparision Schemes To highlight the advantages of the proposed MS-FL architecture, we benchmark it against two representative baselines: • Cloud-based FL (hierarchical FL)—following the layered design in [10], N edge servers serve as intermediaries between vehicles and a single cloud server.
- By enabling neighboring servers to exchange model updates directly—rather than routing everything through a central cloud—MS-FL achieves faster convergence and superior predictive performance compared to both single-server FL and hierarchical edge cloud-based FL.
- Vehicles’ Models’ Aggregation at Server For simplicity, we assume that the Federated Averaging (FedAvg) [5,6] algorithm is employed at each server to efficiently merge local updates from vehicles; however, other client aggregation approaches would work as well.
- Performance Evaluation of the Proposed Server-Level Aggregation Methods To benchmark our proposed MS-FL framework against the vanilla FL setup [5]—which does not share knowledge between servers—we adopted the same shard-based non-IID distribution from [5].

---

## futureinternet-17-00343-v2.pdf

### Summary
This study presents a lightweight autoencoder-based approach for anomaly detection in digit recognition using federated learning on resource-constrained embedded devices. We implement and evaluate compact autoencoder models on the ESP32-CAM microcontroller, enabling both training and inference directly on the device using 32-bit floating-point arithmetic. The system is trained on a reduced MNIST dataset (1000 resized samples) and evaluated using EMNIST and MNIST-C for anomaly detection. Seven fully connected autoencoder architectures are first evaluated on a PC to explore the impact of model size and batch size on training time and anomaly detection performance. Selected models are then re-implemented in the C programming language and deployed on a single ESP32 device, achieving training times as short as 12 min, inference latency as low as 9 ms, and F1 scores of up to 0.87. Autoencoders are further tested on ten devices in a real-world federated learning experiment using Wi-Fi. We exp...

### Model Architecture & Pipeline
- Instead of hard-coding a specific AE design, the ESP32-CAM establishes a Wi-Fi connection to a central server, which provides a complete training configuration, including the model architecture, early stopping Future Internet 2025, 17, 343 9 of 34 parameters, activation functions, optimizer type, and other training-related parameters.
- Table 5 summarizes the results for all seven models, including the model architecture, batch size, number of epochs until convergence (based on a dual-phase early stopping criterion), final training loss, total training time, inference time, AD rates, and F1 score.
- Autoencoder Deployment on ESP32-CAM This section presents the deployment of an autoencoder on the ESP32 MCU, covering dataset selection and modification, the model architecture and training strategy, and the em- bedded implementation, including on-device training.
- Evaluation of AutoencoderPerformance on PC Before deploying autoencoder models to embedded devices, we evaluated seven architectures on a PC to understand how model complexity and batch size influence both training efficiency and AD performance.
- At the same time, the growing concern for data privacy [11] has prompted the ex- ploration of federated learning (FL) [12], a collaborative approach in which devices train models locally and share only model updates rather than raw data.

### Accuracies / Metrics Achieved
- Model evaluation was conducted using several performance metrics, including final training loss (measured by MSE), training and inference time, AD rate (defined as the percentage of test samples flagged as anomalous), and the F1 score based on AD results.
- AD was performed by measuring the reconstruction error using mean squared error (MSE) between the input and the reconstructed output image as the MSE provides a standard and effective metric for quantifying reconstruction discrepancies in image data.
- Integer arithmetic was employed wherever possible to reduce memory consumption and increase execution speed, while model weights were kept in floating-point format to maintain accuracy comparable to standard TensorFlow implementations.
- TinyTrain [31] introduces a task-adaptive sparse update mechanism that selectively fine-tunes layers or channels, significantly reducing memory and computation overhead while maintaining accuracy and fast training.
- Although the ESP32-CAM showed slightly lower accuracy for larger models due to the previously mentioned precision, performance trends remained consistent, validating the robustness of the embedded implementation.

### Baselines & Benchmarks
- As the number of devices trained on fog increases, their models have greater difficulty reconstructing EMNIST digits, leading to higher variability in F1-EMNIST scores compared to MNIST-biased models, which perform better at reconstructing EMNIST.
- After each training round, devices send their updated models to the server, which aggregates them using federated averaging (FedAvg) [52] and redistributes the resulting global model to all participants for the next round.
- The table also includes F1 scores along with their standard deviations across the devices, showing that the variation is larger after local training rounds compared to federated rounds, which aligns with expectations.
- Dataset Allocation in PSRAM The training dataset is stored in external PSRAM due to its larger capacity and faster access compared to internal Flash or SD cards, as demonstrated in our previous work [57].
- 0277 - - - - - - Future Internet 2025, 17, 343 29 of 34 In contrast, our study evaluates the inherent resilience of the standard FedAvg aggre- gation method, which averages model updates from all devices.

---

## futureinternet-17-00409.pdf

### Summary
The Industrial Internet of Things (IIoT) is transforming industrial operations through connected devices and real-time automation but also introduces significant cybersecurity risks. Cyber threat intelligence (CTI) is critical for detecting and mitigating such threats, yet traditional centralized CTI approaches face limitations in latency, scalability, and data privacy. Federated learning (FL) offers a privacy-preserving alternative by enabling de- centralized model training without sharing raw data. This survey explores how FL can enhance CTI in IIoT environments. It reviews FL architectures, orchestration strategies, and aggregation methods, and maps their applications to domains such as intrusion detection, malware analysis, botnet mitigation, anomaly detection, and trust management. Among its contributions is an empirical synthesis comparing FL aggregation strategies—including FedAvg, FedProx, Krum, ClippedAvg, and Multi-Krum—across accuracy, robustness, and efficiency under IIoT c...

### Model Architecture & Pipeline
- Empirical Comparisons: A Strategic Lens on Aggregation Trade-Offs A key contribution of this survey is the synthesis of empirical findings that reveal critical trade-offs among federated learning (FL) aggregation methods within the context of cyber threat intelligence (CTI) for Industrial IoT (IIoT) environments.
- Future Research Directions Future research in federated learning (FL) for cybersecurity and cyber threat intelli- gence (CTI) in Industrial Internet of Things (IIoT) environments must address systemic limitations, contextual deployment constraints, and evolving adversarial threats.
- Recent FL-based approaches enhance phishing and spam detection by combining se- cure training with natural language processing (NLP), lightweight model architectures, and robust aggregation methods to address IIoT-specific challenges such as non-IID data and adversarial risks.
- [65] propose a DNN-based flowchart for distributed traffic classification, while [66] use graph-based models (Fed-MalGAT, Fed-MalGCN) to leverage function call graphs (FCGs) and capture semantic code structures—improving detection accuracy in complex threat scenarios.
- [44] used FedAvg in a deep learning framework for zero-day botnet detection but encountered limitations under non-IID data and adversar- ial interference—highlighting a need for robust aggregation, such as Krum or ClippedAvg, as discussed in Section 3.

### Accuracies / Metrics Achieved
- [65] propose a DNN-based flowchart for distributed traffic classification, while [66] use graph-based models (Fed-MalGAT, Fed-MalGCN) to leverage function call graphs (FCGs) and capture semantic code structures—improving detection accuracy in complex threat scenarios.
- • We contribute an empirical synthesis of FL aggregation strategies outlining trade-offs in accuracy, robustness, and efficiency under IIoT-specific constraints— offering a practical decision-support tool for system designers.
- This analysis offers practical insights into the trade-offs between accuracy, robustness, convergence speed, and computational cost—guiding system designers in selecting appropriate techniques based on deployment requirements.
- Among its contributions is an empirical synthesis comparing FL aggregation strategies—including FedAvg, FedProx, Krum, ClippedAvg, and Multi-Krum—across accuracy, robustness, and efficiency under IIoT constraints.
- FL-based models can reach 90% accuracy in IID settings, but this can drop by 20% under non-IID distributions—an issue highly relevant for CTI in IIoT, where data are often sparse, imbalanced, and decentralized.

### Baselines & Benchmarks
- This structured roadmap ensures that near-term research addresses immediate practi- cal gaps (lightweight models, benchmarks, testbeds), while medium-term efforts focus on interoperability and resilience, and long-term priorities target quantum-era security and fully autonomous FL-CTI ecosystems.
- [44] used FedAvg in a deep learning framework for zero-day botnet detection but encountered limitations under non-IID data and adversar- ial interference—highlighting a need for robust aggregation, such as Krum or ClippedAvg, as discussed in Section 3.
- Approach Description Advantages Disadvantages Representative Use Case/Reference FedAvg (Average) Mean of client updates [23] Simple, effective Weak under highly non-IID data Standard baseline for FL, widely used in CTI frameworks [29,36] Clipped Avg.
- For example, while FedAvg remains widely adopted due to its simplicity and low computational cost, several studies highlight its vul- nerability under highly non-IID conditions and susceptibility to poisoning attacks [29,44].
- Selecting aggregation strategies suited to the threat model and system constraints—such as ClippedAvg for robustness or FedAvg for low-resource scenarios—is critical for maintaining performance in real-world deployments.

---

## futureinternet-17-00492.pdf

### Summary
The rapid proliferation of devices on the Internet of Things (IoT) in smart city environments enables autonomous decision-making, but introduces challenges of scalability, coordina- tion, and privacy. Existing reinforcement learning (RL) methods, such as Multi-Agent Actor–Critic (MAAC), depend on centralized critics and recurrent structures, which limit scalability and create single points of failure. This paper proposes a Federated Decision Transformer (FDT) framework that integrates transformer-based sequence modeling with federated learning. By replacing centralized critics with self-attention-driven trajectory modeling, the FDT preserves data locality, enhances privacy, and supports decentralized policy learning across distributed IoT nodes. We benchmarked the FDT against MAAC in a mobile edge computing (MEC) environment with identical hyperparameter configura- tions. The results demonstrate that the FDT achieves superior reward efficiency, scalability, and adaptability in dynamic ...

### Model Architecture & Pipeline
- Although other attention-based MARL methods, such as Actor– Attention–Critic (AAC) [12], the multi-agent transformer (MAT) [21], and the Multi-Agent Decision Transformer (MADT) [22], represent important advances; they are considered Future Internet 2025, 17, 492 7 of 14 here at a conceptual level rather than re-implemented.
- Article Federated Decision Transformers for Scalable Reinforcement Learning in Smart City IoT Systems Laila AlTerkawi * and Mokhled AlTarawneh Computer Engineering and Cybersecurity Department, College of Engineering and Computing, International University of Kuwait (IUK), Ardiya 92400, Kuwait; mokhled.
- RL offers a promising framework for such tasks, yet traditional actor–critic and value-based approaches struggle with two persistent challenges: capturing long-term temporal dependencies and scaling across large heterogeneous IoT networks [8–11].
- The proposed methodology integrates client-side Decision Transformer training with server-side federated aggregation, enabling scalable and privacy-preserving reinforcement learning across heterogeneous IoT environments.
- Algorithms 1 and 2 formalize the decen- tralized workflow, while complexity analysis highlights the trade-offs between computation (O(L2dH) self-attention per sequence) and communication (O(|θ|) parameters per upload).

### Accuracies / Metrics Achieved
- In the following sections, we detail our experimental setup, evaluation metrics (deci- sion accuracy, convergence speed, communication overhead, and security robustness), and comparative results against MAAC baselines.
- Performance was assessed in terms of decision accuracy, convergence speed, scalability, communication overhead, and robustness.

### Baselines & Benchmarks
- Future work will extend the FDT with variance-reduction strategies, modular aggre- gation schemes, and real-world benchmarking on traffic and energy datasets to further validate its generalizability and deployment readiness in smart city environments [47–49].
- We empirically benchmarked the FDT against MAAC in mobile edge computing simulations, showing improved reward efficiency, adaptability, and scalability, with a trade-off of slightly higher variance during early training [35,36].
- In the following sections, we detail our experimental setup, evaluation metrics (deci- sion accuracy, convergence speed, communication overhead, and security robustness), and comparative results against MAAC baselines.
- Baseline Comparison Scope Our quantitative evaluation focuses on the MAAC baseline, implemented using the publicly available codebase of the authors to ensure reproducibility with identical hyperparameter settings.
- In multi-agent reinforcement learning (MARL), attention-based designs such as the multi- agent transformer (MAT) [21] have achieved state-of-the-art performance in cooperative benchmarks.

---

## futureinternet-17-00505.pdf

### Summary
Federated learning (FL) is a foundational technology for enabling collaborative intelligence in vehicular edge computing (VEC). However, the volatile network topology caused by high vehicle mobility and the profound security risks of model poisoning attacks severely undermine its practical deployment. This paper introduces DTB-FL, a novel framework that synergistically integrates digital twin (DT) and blockchain technologies to establish a secure and efficient learning paradigm. DTB-FL leverages a digital twin to create a real-time virtual replica of the network, enabling a predictive, mobility-aware participant selection strategy that preemptively mitigates network instability. Concurrently, a private blockchain underpins a decentralized trust infrastructure, employing a dynamic reputation system to secure model aggregation and smart contracts to automate fair incentives. Crucially, these components are synergistic: The DT provides a stable cohort of participants, enhancing the accura...

### Model Architecture & Pipeline
- Recent work focuses on making blockchain–FL more scalable and trust-aware: Sharding reduces on-chain bottlenecks [18]; surveys and system designs consolidate best practices for privacy, accountability, and fairness [33,34]; privacy-preserving com- putation with homomorphic encryption is being optimized for FL pipelines [19].
- The remainder of this paper is organized as follows: Section 2 reviews related work; Section 3 presents the system model and problem formulation; Section 4 details the DTB-FL framework design; Section 5 analyzes convergence and complexity; Section 6 evaluates performance through simulations; Section 7 concludes this paper.
- DTB-FL fills this gap through a unified architecture where DTs enable proactive partic- ipant profiling and network forecasting, blockchain provides tamper-evident validation and programmable incentives, and FL leverages these integrated signals to achieve secure and efficient training under realistic VEC constraints.
- Threat Model We consider a realistic threat model that reflects the security challenges in vehicular federated learning: Future Internet 2023, 17, 505 9 of 37 Trusted Entities: The edge server (ES) and roadside units (RSUs) are assumed to be trusted infrastructure components operated by reliable network providers.
- This formulation captures the fundamental trade-offs in vehicular federated learning: The system must balance training efficiency (minimizing time), operational cost (minimiz- ing rewards), model quality (meeting accuracy targets), and security (limiting adversarial influence).

### Accuracies / Metrics Achieved
- The synergy emerges because each technology addresses a gap in the others: DT handles dynamic uncertainty (mobility and resources), BC handles trust uncertainty (malicious actors and incentives), and FL provides a collaborative learning substrate that benefits from both while feeding back performance data to improve their accuracy.
- This formulation captures the fundamental trade-offs in vehicular federated learning: The system must balance training efficiency (minimizing time), operational cost (minimiz- ing rewards), model quality (meeting accuracy targets), and security (limiting adversarial influence).
- While the optimization algorithm itself is the standard FedAvg, the quality and trustworthi- ness of the aggregated updates are substantially improved through intelligent selection and reputation-based filtering, leading to superior convergence speed and final accuracy.
- Problem Formulation Building on the models presented above, we formulate a multi-objective optimiza- tion problem that minimizes the system operator’s total cost (comprising training time and reward payments) while maintaining model accuracy and security guarantees.
- Extensive simulations demonstrate that DTB-FL accelerates model convergence by 43% compared to FedAvg and maintains 75% accuracy under poisoning attacks even when 40% of participants are malicious—a scenario where baseline FL methods degrade to below 40% accuracy.

### Baselines & Benchmarks
- Class 0 1 2 3 4 5 6 7 8 9 Rounds 52 48 51 47 53 49 50 46 52 48 While our current utility function does not explicitly optimize for data diversity, the empirical results demonstrate that natural diversity is maintained in practice due to the dynamic nature of vehicular environments and the relatively large selection size (K = 10) compared to the number of classes.
- DTB-FL’s per-round time is higher than simpler baselines due to two additional components: (1) DT-based prediction for intelligent participant selection (gap from VEC-FL to DT-FL) and (2) blockchain operations for reputation management and secure aggregation (gap from DT-FL to DTB-FL).
- While the optimization algorithm itself is the standard FedAvg, the quality and trustworthi- ness of the aggregated updates are substantially improved through intelligent selection and reputation-based filtering, leading to superior convergence speed and final accuracy.
- Extensive simulations demonstrate that DTB-FL accelerates model convergence by 43% compared to FedAvg and maintains 75% accuracy under poisoning attacks even when 40% of participants are malicious—a scenario where baseline FL methods degrade to below 40% accuracy.
- Comparison Baselines To demonstrate the superiority of DTB-FL, we compare it against the following baselines: • FedAvg [47]: The standard FL algorithm, where the server randomly selects partici- pants and performs simple weighted averaging.

---

## ijgi-13-00210-v3.pdf

### Summary
In response to the insufficient consideration of spatio-temporal dependencies and traffic pattern similarity in traffic flow prediction methods based on federated learning, as well as the neglect of model heterogeneity and objective heterogeneity, a traffic flow prediction model based on federated learning and spatio-temporal graph neural networks is proposed. The model is divided into two stages. In the road network division stage, the traffic road network is divided into subnetworks by the dynamic time warping algorithm and the K-means algorithm, to ensure the same subnetwork has the similar traffic flow pattern. The federated learning stage is divided into two sub-stages. In the local training phase, the spatio-temporal graph neural network with an attention mechanism is utilized to create personalized models and meme models to capture the spatio-temporal dependencies of each subnetwork. At the same time, deep mutual learning is utilized to address model heterogeneity and objective ...

### Model Architecture & Pipeline
- cn Abstract: In response to the insufficient consideration of spatio-temporal dependencies and traffic pattern similarity in traffic flow prediction methods based on federated learning, as well as the neglect of model heterogeneity and objective heterogeneity, a traffic flow prediction model based on federated learning and spatio-temporal graph neural networks is proposed.
- Ablation Experiment for PSS To validate the effectiveness of the PSS, strategies for road network division adopted by DST-GCN and the FCGCN are used to form two contrasting models separately, including FedTFP-R, dividing the road network randomly according to a proportion, and FedTFP-L, dividing the road network by the Louvain algorithm.
- eij = wxi t · wxj t (2) aij = so f tmax  eij  ⊙Ack (3) xi t = ∑ j∈Ni aij · wxj t + b (4) where eij denotes the attention coefficients, w is the weight matrix, aij is the result of normalization for the attention coefficients and ⊙denotes the Hadamard product.
- Lo = MSE(y, AFSTGCN(Ack,Sck)) (7) Lc = MSE(y, AFSTGCNk(Ack,Sck)) (8) La = MSE(AFSTGCNk(Ack,Sck), AFSTGCN(Ack,Sck)) (9) Lb = MSE(AFSTGCN(Ack,Sck), AFSTGCNk(Ack,Sck)) (10) Lm = βLo + (1 −β)La (11) Lp = αLc + (1 −α)Lb (12) where MSE is the Mean Squared Error.
- The main contributions are as follows: (1) A new model FedTFP for traffic flow prediction based on FL and AFSTGCN is proposed to protect data privacy as well as learn the spatio-temporal dependencies of traffic flow comprehensively.

### Accuracies / Metrics Achieved
- Lo = MSE(y, AFSTGCN(Ack,Sck)) (7) Lc = MSE(y, AFSTGCNk(Ack,Sck)) (8) La = MSE(AFSTGCNk(Ack,Sck), AFSTGCN(Ack,Sck)) (9) Lb = MSE(AFSTGCN(Ack,Sck), AFSTGCNk(Ack,Sck)) (10) Lm = βLo + (1 −β)La (11) Lp = αLc + (1 −α)Lb (12) where MSE is the Mean Squared Error.
- Evaluation Metrics Mean Absolute Error (MAE) [29], Mean Absolute Percentage Error (MAPE) [29] and Root Mean Square Error (RMSE) [29] are used to evaluate the prediction performance of FedTFP.
- As shown in Figures 5–7, MAE and RMSE values decrease first and then increase with the increase in γ, and the MAPE does not change first and then increases.
- Firstly, after the training of the local model, the model performance MSE of the k-th client is obtained as pk.
- Dataset Client MAE RMSE MAPE PeMS04 Client 0 18.

### Baselines & Benchmarks
- Ablation Experiment for PSS To validate the effectiveness of the PSS, strategies for road network division adopted by DST-GCN and the FCGCN are used to form two contrasting models separately, including FedTFP-R, dividing the road network randomly according to a proportion, and FedTFP-L, dividing the road network by the Louvain algorithm.
- Lo = MSE(y, AFSTGCN(Ack,Sck)) (7) Lc = MSE(y, AFSTGCNk(Ack,Sck)) (8) La = MSE(AFSTGCNk(Ack,Sck), AFSTGCN(Ack,Sck)) (9) Lb = MSE(AFSTGCN(Ack,Sck), AFSTGCNk(Ack,Sck)) (10) Lm = βLo + (1 −β)La (11) Lp = αLc + (1 −α)Lb (12) where MSE is the Mean Squared Error.
- The main contributions are as follows: (1) A new model FedTFP for traffic flow prediction based on FL and AFSTGCN is proposed to protect data privacy as well as learn the spatio-temporal dependencies of traffic flow comprehensively.
- In the local training phase, personalized models and meme models on local clients are created based on STGNNs with an attention mechanism (AF- STGCN) to achieve personalized learning and global aggregation separately.
- DST-GCN divides the road network evenly based on the proportion of the dataset, and the FCGCN employs the Louvain algorithm to divide the road network based on the connection patterns between road nodes.

---

## information-16-00861-v2.pdf

### Summary
Non-IID is one of the key challenges in federated learning. Data heterogeneity may lead to slower convergence, reduced accuracy, and more training rounds. To address the common Non-IID data distribution problem in federated learning, we propose a comprehensive dynamic optimization approach based on existing methods. It leverages MAP estimation of the Dirichlet parameter β to dynamically adjust the regularization coefficient µ and introduces orthogonal gradient coefficients ∆i to mitigate gradient interference among different classes. The approach is compatible with existing federated learning frameworks and can be easily integrated. Achieves significant accuracy improvements in both mildly and severely Non-IID scenarios while maintaining a strong performance lower bound.

### Model Architecture & Pipeline
- System Model In this section, we provide a detailed description of the system model, including the federated learning architecture, The estimation of the degree of Non-IID using the MAP is based on the number of classes in the local dataset, the coefficient of the regularization term µ in the objective function, and the orthogonal tensor coefficients in the local updating stage.
- In the work [31], focusing on spatial behavior and drawing inspiration from neuromod- ulation, proposed a framework with three distinct granularities: plastic neurons based on Hebbian learning (fine-grained), layers with dropout (medium-grained), and network layers with self-regulating learning rates (coarse-grained).
- In recent years, researchers have proposed various heuristic algorithms inspired by biological intelligence, including mechanisms that mimic memory and synaptic consolida- tion, to address catastrophic forgetting [24,25], attracting significant attention.
- The proximal term µ in the objective function significantly affects the performance of federated learning, with larger values typically assigned under severe Non-IID conditions and smaller values under near-IID conditions.
- Introduction Federated Learning (FL) [1,2] is an emerging distributed machine learning framework that enables knowledge sharing and user privacy protection during model training without requiring the upload of raw data.

### Accuracies / Metrics Achieved
- 0 20% 30% 40% Target Accuracy 0 20 40 60 80 100 Rounds Mild Non-IID 20% 30% 40% Target Accuracy 0 20 40 60 80 100 Rounds 20% 30% 40% Target Accuracy 0 20 40 60 80 100 Rounds 20% 30% 40% Target Accuracy 0 20 40 60 80 100 Rounds IID 20% 30% 40% Target Accuracy 0 20 40 60 80 100 Rounds 20% 30% 40% Target Accuracy 0 20 40 60 80 100 Rounds fednaca Proposal fedavg fedprox moon scaffold Figure 11.
- In each communication round, a fraction of clients, referred to as the sampling ratio 0 < C ≤1, is randomly selected to participate in model training; while both metrics capture aspects of convergence, the number of rounds to reach the target accuracy provides a more intuitive and direct measure of convergence speed across different algorithms compared to tracking accuracy curves.
- Results on Top-1 Accuracy The number of communication rounds required to reach the target accuracy serves as a measure of the convergence speed of each approach, whereas the top-1 accuracy reflects the model’s reliability.
- Extensive experimental results demonstrate that our approach achieves significant accuracy improvements in both mildly and severely Non-IID scenarios while maintaining a strong performance lower bound.
- Metrics To evaluate the convergence speed of different methods, we employ two metrics: (1) the number of communication rounds required to reach a predefined target accuracy and (2) the top-1 accuracy.

### Baselines & Benchmarks
- In each communication round, a fraction of clients, referred to as the sampling ratio 0 < C ≤1, is randomly selected to participate in model training; while both metrics capture aspects of convergence, the number of rounds to reach the target accuracy provides a more intuitive and direct measure of convergence speed across different algorithms compared to tracking accuracy curves.
- The proposed method is expected to achieve performance comparable to advanced approaches such as FedProx under IID settings, and to brain-inspired approaches such as FedNACA under Non-IID settings, as illustrated by the blue dashed line in Figure 1.
- Inspired by the memory system of Drosophila, the study [30] proposed a solution that incorporates two additional modules compared to traditional artificial intelligence: stability protection and active forgetting.
- Some approaches, such as FedAvg, exhibit poor performance under conditions of low client sampling ratios and severe Non-IID data distributions, as illustrated in the first subplot of the first row in Figure 10.
- Extensive exper- iments conducted on benchmark datasets demonstrate the effectiveness and robustness of the proposed method in comparison with classical baselines under varying degrees of data heterogeneity.

---

## International Transactions on Electrical Energy Systems - 2025 - Lin - Active Privacy‐Preserving  Distributed Edge Cloud.pdf

### Summary
Research Article Active Privacy-Preserving, Distributed Edge–Cloud Orchestration–Empowered Smart Residential Mains Energy Disaggregation in Horizontal Federated Learning Yu-Hsiu Lin ,1,2 Yung-Yao Chen ,3 and Shih-Hao Wei1 1Graduate Institute of Automation Technology, National Taipei University of Technology, Taipei 106344, Taiwan 2Research Center of Energy Conservation for New Generation of Residential, Commercial, and Industrial Sectors, National Taipei University of Technology, Taipei 106344, Taiwan 3Department of Electronic and Computer Engineering, National Taiwan University of Science and Technology, Taipei 106335, Taiwan Correspondence should be addressed to Yu-Hsiu Lin; yhlin@ntut.edu.tw Received 23 October 2024; Accepted 28 February 2025 Academic Editor: Murthy Cherukuri Copyright © 2025 Yu-Hsiu Lin et al. International Transactions on Electrical Energy Systems published by John Wiley & Sons Ltd. Tis is an open access article under the terms of the Creative Commons Attribution ...

### Model Architecture & Pipeline
- In the developed framework, by cooperating with the cloud to achieve consolidated global AI in an edge–cloud collaborative computing fashion, the edge computing paradigm with improved performance provides converged computing at the edge, through which local private sensory load data gathered from distributed on-site autonomous edge devices are leveraged through global AI in HFL.
- Ad- ditionally, trainingless/seamless/fully nonintrusive online autonomous energy disaggregation in the feld of Client C, which is newly contained in the framework, can be seam- lessly achieved by the global consolidated AI model that has already been trained over local private load data of dis- tributed Clients A and B without the data security and privacy concerns.
- Simulation Results In this section, the proposed energy management frame- work, that is, an active privacy-preserving and distributed edge–cloud collaborative computing–based energy man- agement framework, for implementing smart residential mains energy disaggregation at the edge is demonstrated and evaluated through simulations conducted in a laboratory environment.
- In this regard, in [9], an energy disaggregation approach based on an event-driven factorial hidden Markov model was implemented in an edge–cloud computing framework, where data-intensive event detection was performed at the edge side and model-intensive load disaggregation was completed at the cloud side (the cloud server took over the task of load disaggregation).
- In addition, the presented scheme with the time-series load modeling and forecasting mechanism in [20] can be conducted to, integrated with, and used by the residential community energy management framework in [47] to autonomously and nonintrusively parse required data like appliance energy consumption data and users’ preferences for using diferent appliances.

### Accuracies / Metrics Achieved
- Accuracy (%) Improvement (%) Without active FedNILM With active FedNILM Client A 80.
- 99 1223 Note: Te overall classifcation accuracy is 0.
- 80 Accuracy (global) 0.

### Baselines & Benchmarks
- Finally, the presented scheme, based on FedAVG, to be commercialized will be implemented in either the WISE-PaaS/AIFS (AI Frame- work Service) solution or FedML (FedML is an open re- search library and benchmark to (1) build ML models based on distributed datasets over multiple locations and (2) commercialize FL easily, scalably, and economically).
- In the meanwhile, it can be based on FedProx [48], a version of generalization and reparametrization of FedAVG, to be allowed for more robust convergence than that of FedAVG practically.
- ai), which is an open research library and benchmark to commercialize FL easily, scalably, and economically.
- A comparative summary of energy disaggregation compared to ILM can be found in [8, 11].

---

## mathematics-12-02539.pdf

### Summary
Wireless traffic prediction is essential to developing intelligent communication networks that facilitate efficient resource allocation. Along this line, decentralized wireless traffic prediction under the paradigm of federated learning is becoming increasingly significant. Compared to tradi- tional centralized learning, federated learning satisfies network operators’ requirements for sensitive data protection and reduces the consumption of network resources. In this paper, we propose a novel communication-efficient federated learning framework, named FedCE, by developing a gradient compression scheme and an adaptive aggregation strategy for wireless traffic prediction. FedCE achieves gradient compression through top-K sparsification and can largely relieve the communi- cation burdens between local clients and the central server, making it communication-efficient. An adaptive aggregation strategy is designed by quantifying the different contributions of local models to the global model...

### Model Architecture & Pipeline
- In addition, a new federated learning algorithm is proposed that fully leverages variance reduction methods, utilizing local and global control variables on both the local device and the central server to track local updates and address the issue of client drift [14].
- The concept of federated learning was first proposed by the Google [8], which allows multiple local devices to store unevenly distributed data and collaboratively train high- quality centralized models while protecting data privacy.
- A new federated framework called FedLoc is proposed, which successfully collects a smaller local dataset without sacrificing user privacy and approximates the global machine learning model in a collaborative manner [20].
- Furthermore, since federated learning only receives gradient information, performing spatial–temporal dependencies analysis on the original data is impossible, which negatively impacts the model’s prediction accuracy.
- The same lightweight MLP network architecture [22] is used for the experiment to reduce the influence of other factors in FL, which can reduce computation and storage costs while maintaining high prediction accuracy.

### Accuracies / Metrics Achieved
- Overall, the use of CDRs from Milan and Trentito as the primary datasets, along with the evaluation metrics of MAE and MSE, allowed us to conduct a rigorous and comprehensive analysis of their prediction model’s performance.
- Furthermore, since federated learning only receives gradient information, performing spatial–temporal dependencies analysis on the original data is impossible, which negatively impacts the model’s prediction accuracy.
- The same lightweight MLP network architecture [22] is used for the experiment to reduce the influence of other factors in FL, which can reduce computation and storage costs while maintaining high prediction accuracy.
- However, with the increasing contradiction between sensitive data leakage and the accuracy of wireless traffic prediction, federated learning-based wireless traffic prediction has received more attention [19].
- Specifically, while FedCE demonstrates comparable accuracy to the baseline algorithms on the Milano dataset, its predictive precision on the Trento dataset significantly surpasses that of its counterparts.

### Baselines & Benchmarks
- Within these figures, the actual observed values are denoted by a solid blue line, whilst the forecasted values for FedCE, FedDA, and FedAvg are represented by a green dashed line, an orange dash-dot line, and a red dash-dot line, respectively.
- On the Trento dataset, FedCE demonstrates the highest prediction performance, fol- lowed by other baseline algorithms in descending order: centralized training methods, FedDA, standalone training methods, and FedAvg.
- Conversely, Figure 4b highlights that during peak traffic intervals, FedCE’s forecasted values more closely approximate the ground truth than those Mathematics 2024, 12, 2539 12 of 14 of the baseline algorithms.
- Based on this, the proposed FedCE method was compared with four baseline methods from three different perspectives, centralized, fully distributed, and federated, to verify its effectiveness and practicality.
- Specifically, while FedCE demonstrates comparable accuracy to the baseline algorithms on the Milano dataset, its predictive precision on the Trento dataset significantly surpasses that of its counterparts.

---

## Position-Aware_Structural_Knowledge_Sharing-Based_Federated_Graph_Learning_for_Intelligent_Transportation_Systems.pdf

### Summary
Federated Graph Learning (FGL) combines the powerful graph data modeling capabilities of Graph Neural Networks (GNNs) with the distributed processing requirements of intelligent transportation systems (ITS), making it a prominent focus in recent research. In the context of the Internet of Everything (IoE), diverse devices and services generate highly heterogeneous, non-independent, and non-identically distributed (non-IID) data, which limits model generalization and train- ing efficiency. To tackle these challenges, this paper proposes a Position-Aware Structural Sharing Federated Graph Learn- ing Framework tailored for ITS applications. This framework enhances GNNs’ capacity to process cross-domain graph data, significantly improving model applicability and performance across various ITS scenarios. Specifically, we use a structural encoder alongside a position-aware structural encoder to capture generic structural knowledge, sharing these embeddings across clients in an FGL setup. Ext...

### Model Architecture & Pipeline
- This feature-structure separation architecture ensures that the network can capture both attribute feature of the nodes and their structural positions in graph, and the model can extract useful graph structural knowledge while retaining each domain-specific information, enhancing its adaptability and prediction performance on diverse data sources.
- 0 as recommended in their work; FedStar [6], which addresses non-IID data through feature-structure decoupling; and FED-PUB [32], which pro- poses a personalized subgraph federated learning framework based on functional embedding, calculating model similarity through random graph input and combining sparse masks to achieve parameter localization.
- The goal of the federated learning framework is to optimize the global objective function as follows: min (w1,w2,··· ,wM) 1 M M X m=1 |Dm| N Lm (wm; Dm) , (1) Here, N denotes the total number of all client data instances, and Lm and wm represent the loss function and model parameters of the m-th client, respectively.
- embedding, together with the global position-aware structural embedding captured through the position-aware structural encoder to represent the global structural information, and this approach helps different clients in federated learning to better understand the global structural information.
- The overall layout of our proposed GNN architecture: the the blue colored boxes show models that are locally trained and customized for each client, and the green boxes represent models that are aggregated on the server-side (which allows for the sharing of knowledge across clients).

### Accuracies / Metrics Achieved
- However, the strategy of randomly selecting anchor points in P-GNN may ignore some important graph structural features or node relationships, which will affect the model’s in-depth understanding and accuracy of the graph data.
- Accuracy graphs for our method and three FGL approaches over communication rounds, with subplots for unique cross-dataset (a) or cross-domain (b,c,d) non-IID conditions.
- Our method demonstrates superior convergence properties in all scenarios, achieving faster convergence and higher accuracy compared to baseline methods.
- 4, we ana- lyze the convergence behavior across four non-IID scenarios through test accuracy curves averaged over randomized trials.
- 90% accuracy with a 4.

### Baselines & Benchmarks
- The narrow standard deviation bands in the convergence curves demonstrate that our method achieves both faster convergence and better stability compared to baseline approaches, highlighting its scalability in handling non-IID graph data across different domains.
- The basic federated learning methods include Local, where clients train independently without communication; FedAvg [14], the standard federated averaging algorithm; FedProx [16], which addresses client heterogeneity through proximal regularization (µ = 0.
- FedAvg is a prime example of such an approach, which is implemented by periodically merging model parameters from all clients on a server: w ← M X m=1 |Dm| N wm, (2) This merged average model is then distributed back to each client.
- : POSITION-AWARE STRUCTURAL KNOWLEDGE SHARING-BASED FGL FOR ITS 9 TABLE I PERFORMANCE IS EVALUATED ACROSS VARIOUS FEDERATED GRAPH CLASSIFICATION SCENARIOS, WHERE EACH SCENARIO ENCOMPASSES DISTINCT DATASETS HELD BY SEPARATE CLIENTS.
- Experimental Results and Analysis 1) Performance Comparison: Extensive experiments were conducted across four non-IID scenarios to evaluate our proposed method against baseline approaches, with results summarized in Table I.

---

## s41598-025-24963-z.pdf

### Summary
FedGDAN: Privacy-preserving traffic flow prediction via federated graph diffusion attention networks Yuanhui Li1, Bo Mi2 & Ran Zeng2 Efficient data utilization and strong privacy protection are major challenges in Intelligent Transportation Systems (ITS), particularly in complex environments with highly distributed Intelligent Connected Vehicles (ICVs). Conventional machine learning methods struggle to capture complex spatiotemporal dependencies while maintaining data privacy and locality. To overcome these limitations, we propose FedGDAN, a Federated Graph Diffusion Attention Network that combines graph neural networks (GNNs) with federated learning (FL) to enable collaborative traffic flow prediction without sharing raw data. FedGDAN models global spatiotemporal correlations across road networks and introduces an adaptive local aggregation mechanism to address non-independent and identically data distributions, thereby improving robustness and accuracy. Experiments on real-world dat...

### Model Architecture & Pipeline
- FedGDAN: Privacy-preserving traffic flow prediction via federated graph diffusion attention networks Yuanhui Li1, Bo Mi2 & Ran Zeng2 Efficient data utilization and strong privacy protection are major challenges in Intelligent Transportation Systems (ITS), particularly in complex environments with highly distributed Intelligent Connected Vehicles (ICVs).
- Unlike prior approaches that either focus solely on privacy13,25 or assume IID data distributions26–28, FedGDAN integrates a graph diffusion attention mechanism with adaptive local aggregation to simultaneously preserve spatiotemporal data and road network topology, while effectively addressing the Non-IID data problem across different clients.
- Specifically, the propagation rule for node representations at each layer of a GCN is as follows: Hk+1 = σ ( ∼ D −1 2 ∼ A ∼ D −1 2 HkW k )  (2) Here, ∼ A = A + I represents the augmented adjacency matrix of the given undirected graph G , incorporating self-connections to allow nodes to include their own features in the representation updates.
- In each training round, participants utilize the current global model to update their local model parameters: θi t+1 = θt −η∇Li (θt) (4) Methodology This section first introduces a graph diffusion attention-based spatiotemporal graph neural network as the traffic speed prediction model for federated learning clients.
- com/scientificreports/ generator, which produces spatio-temporal embeddings to provide initial spatio-temporal information for the encoder and decoder; and (3) a transfer attention layer, which leverages historical and future STEs to refine the encoder’s output, thereby mitigating the propagation of errors.

### Accuracies / Metrics Achieved
- com/scientificreports/ • In the proposed federated learning (FL) framework, we introduce an Adaptive Local Aggregation (ALA) method to mitigate the Non-IID data problem across clients caused by differing traffic scenarios, thereby improving the accuracy of collaborative distributed prediction.
- To comprehensively evaluate the performance of different methods in traffic flow prediction tasks, three commonly used evaluation metrics are adopted: Mean Absolute Error (MAE), Root Mean Square Error (RMSE), and Mean Absolute Percentage Error (MAPE).
- This method can share global knowledge while adapting to the local data distribution of different nodes, thereby better capturing the diversity and spatiotemporal variability of traffic data and improving the accuracy of traffic flow prediction.
- FedGDAN models global spatiotemporal correlations across road networks and introduces an adaptive local aggregation mechanism to address non-independent and identically data distributions, thereby improving robustness and accuracy.
- To assess the practical viability of FedGDAN in privacy-sensitive applications, we systematically evaluated the trade-off between privacy protection strength and prediction accuracy under different differential privacy parameters.

### Baselines & Benchmarks
- The FedAvg algorithm can be expressed as follows: Θt = ∑ k∈St nk n Θt i (18) where Θt i denotes the local model parameters of client i, St represents the set of clients participating in round t, nk corresponds to the local data volume of client k, n = ∑ k∈St nk defines the aggregate data volume from participating clients in round t, and Θt constitutes the aggregated global model parameters.
- Extensive experiments and ablation studies on three benchmark datasets demonstrate that FedGDAN consistently outperforms state-of-the-art baselines, exhibiting strong scalability, a balanced trade-off under differential privacy, and robust performance across heterogeneous data distributions.
- com/scientificreports/ Experiment results This study presents a comprehensive evaluation of FedGDAN’s performance across three benchmark datasets, covering prediction tasks at 15-minute (short-term), 30-minute (mid-term), and 60-minute (long-term) intervals.
- This unified initialization paradigm essentially ignores the statistical heterogeneity between clients, resulting in a decrease in the accuracy of the FedAvg algorithm under non-IID data distribution37.
- Through systematic comparisons with four state-of-the-art models, our results demonstrate that FedGDAN consistently outperforms the baselines in all experimental settings, as summarized in Table 1.

---

## sensors-25-01116-v3.pdf

### Summary
Intelligent Transport Systems (ITSs) are essential for secure and privacy- preserving communications in Autonomous Vehicles (AVs) and enhance facilities like connectivity and roadside assistance. Earlier research models used for traffic manage- ment compromised user privacy and exposed sensitive data to potential adversaries while handling real-time data from numerous vehicles. This research introduces a Federated Learning-based Predictive Traffic Management (FLPTM) system designed to optimize service access and privacy for Autonomous Vehicles (AVs) within an ITS. Moreover, a CPPS will provide strong security to mitigate adversarial threats through state modelling and authenticated access permissions for the integrity of vehicle communication networks from man-in-the-middle attacks. The suggested FLPTM system utilizes a Contained Privacy- Preserving Scheme (CPPS) that decentralizes data processing and allows vehicles to train local models without sharing raw data. The CPPS framework co...

### Model Architecture & Pipeline
- Trust management on the Internet of Vehicles (IoV), encompassing the importance of trust systems in ensuring secure communication and decision-making in vehicular networks, was examined in [10], which identified the key challenges and solutions for building robust trust management frameworks within IoV ecosystems.
- The proposed FLPTM framework addresses the above challenges by using Federated Learning for decentralized data processing, a contained privacy-preserving scheme for enhanced data security, and robust adversarial resilience mechanisms to guarantee scalable, efficient, and secure real-time traffic management.
- Sensors 2025, 25, 1116 4 of 24 Security Issues: The manuscript also contains a few mechanisms for securing data and communication: • Man-in-the-Middle Protection: the CPPS framework uses authentication protocols, bilinear mapping, and key-based mechanisms to attenuate adversarial threats.
- The proposed framework starts by initializing a list of vehicles, infrastructure, and service request states, followed by the training of local models on vehicle data ρS and infras- Sensors 2025, 25, 1116 9 of 24 tructure data ρI.
- The proposed architecture ensured improved performance and security for vehicular communication systems, highlighting the role of SDN in providing adaptable, centralized control over dynamic vehicular environments [6].

### Accuracies / Metrics Achieved
- The main contributions of this study are as follows: • Predictive modelling for AVs using the FLPTM-CPPS system is applied in this paper, which enables traffic models to be trained locally without centralizing data, thus improving the accuracy of traffic predictions.
- A case study conducted in Hangzhou, China, illustrates how Sensors 2025, 25, 1116 6 of 24 the research idea outperforms conventional models in terms of accuracy in predicting and privacy preservation capacity.
- The two most important goals of an FLPTM system’s localized computation are protecting user privacy and enhancing the estimated accuracy of traffic while guaranteeing efficient and secure communication.
- The suggested method outperforms established methods in terms of accuracy and performance metrics (RMSE, MAE, R2) when tested on a dataset of 4500 cabs in Bangkok using MATLAB2022b.
- In addition, when dealing with traffic patterns particular to regions or constantly changing, the accuracy of traffic analysis models that rely on static data tends to be poorer.

### Baselines & Benchmarks
- Adversary Impact The proposed scheme achieves less adversary impact compared to the other methods.
- The proposed scheme is cost-efficient when compared with the existing schemes.
- ; HAl-Bayatti, A.

---

## S0360835225001500.htm

### Summary
Accurate network traffic prediction is indispensable for efficient load-aware resource management and performance optimization in metro optical networks (MONs). Existing machine learning (ML)-based methods for network node traffic prediction typically adopt a centralized approach, in which a centralized learning model is trained with traffic data collected from all nodes. However, the requirement for transferring massive amount of data to a centralized server in such an approach may increase the communication delays and raise privacy concerns. Although these problems can be solved by using federated learning (FL) technique, the generalization ability and prediction accuracy of a global model in a federated setting are still affected by the diverse traffic patterns and scale of the MON, which are also the limitations of traditional centralized approaches for traffic prediction. To further improve the traffic prediction accuracy in edge computing-enabled MONs, we propose an FL-based traf...

### Model Architecture & Pipeline
- Looked at HTML file.

### Accuracies / Metrics Achieved
- See full text for details.

---

## S2772662224000900.htm

### Summary
A distributed machine learning technique called federated learning allows numerous Internet of Things (IoT) edge devices to work together to train a model without sharing their raw data. Internet of Vehicular Things (IoVT) are an important tool in smart cities for moving objects, such as knowing the traffic patterns, road conditions, vehicle behavior, etc. To enhance traffic management and optimize routes, federated learning, and IoT must work jointly, which may achieve sustainable development goals (SDG) in many ways. This research outlines a system for federated learning in vehicular networks in smart cities. The suggested architecture considers the difficulties presented by such situations’ restricted network connectivity, privacy issues, and security concerns. The framework employs a hybrid methodology integrating federated learning on a centralized server with local training on individual cars. The proposed framework is assessed based on a real-world dataset from a smart city thro...

### Model Architecture & Pipeline
- Looked at HTML file.

### Accuracies / Metrics Achieved
- See full text for details.

---

