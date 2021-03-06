## Experiment on Code Classification

To further explore what is encoded in the learned features, we conducted experiments using a simple one-layer feed-forward neural network with different mechanisms to perform the code classification task.  



#### 1. Dataset

There are 131 different categories in the MTG dataset. Since our model and the comparison methods have only one layer of a feed-forward neural network, whose performance has been much limited, we reduce the number of categories to 5. We selected the five categories with the most data in the MTG dataset and extracted their corresponding codes as the MTG-CC (i.e., MTG for code classification) dataset. The statistics are listed in Table 1. 	  



<center>Table 1: Statistics on MTG-CC dataset.</center>

| Classes               | Train    | Dev    | Test   | Total    |
| --------------------- | -------- | ------ | ------ | -------- |
| Champions of Kamigawa | 248      | 13     | 13     | 274      |
| Shadowmoor            | 241      | 13     | 13     | 267      |
| Innistrad             | 240      | 13     | 13     | 266      |
| Tempest               | 239      | 13     | 13     | 265      |
| Gatecrash             | 211      | 12     | 12     | 235      |
| **MTG-CC**            | **1179** | **64** | **64** | **1307** |


#### 2. Training

The experiment adopts a simple single-layer feedforward neural network with different mechanisms to predict code classes. We chose AST and GraphSAGE as the comparison method for the following reasons. AST contains information about the syntactical structure of the code, and it is a special kind of graph that includes both local structural information and textual information. GraphSAGE is a blueprint for ADG, which can incorporate information about nodes and their neighbours.   

* **AST-based model**    Since the core network of ADG-embedder is LSTM, for reasons of fairness as well as control variables, we used LSTM to learn AST information. We parsed the code into AST and fed it into LSTM. Then, a layer of a feed-forward neural network is connected to predict the category of the code.  
* **GraphSAGE**    We used the GraphSAGE-LSTM as one of the comparison model [1]. In the modeling of the graph, all nodes in the MTG are considered. The sample size is set to 10, and the aggregator adopts LSTM.  
* **ADG-embedder**    The weights in ADG-embedder are inherited from the ADG-Seq2Seq model trained on the MTG dataset. Thus, node embeddings generated by ADG-embedder are fed into the feed-forward neural network. We then utilized a softmax activation function with a learning rate of 0.01 on the cross-entropy loss and Adam optimizer to predict the categories.  



#### 3. Results



<center>Table 2: Code classification results.</center>

| Methods         | Micro-F1 |
| --------------- | -------- |
| AST-based model | 0.603    |
| GraphSAGE-LSTM  | 0.577    |
| ADG-embedder    | 0.652    |



As shown in Table 2, the ADG-based model achieves better results than the other models, indicating that the ADG-embedder is effective even in entirely different tasks, such as code classification.  
There are a number of code representations that utilize code classification to test effectiveness, such as [2], [3] and [4]. 
In Section 7.2.2, we analyzed the API invocation constraints and API call order learned by the ADG embedding, which provides robust supports to code generation. The learned API features are primarily used to generate syntactically correct API calls, e.g., to avoid null pointer exceptions.   
Our model generates embedding vectors that highlight a representation of the semantics inherent to the API itself. Based on the experimental results, we believe that such representation can characterize code fragments. Thus, by developing new specific mechanisms to combine the learned API features with other existing representation models, it is expected to improve the performance of tasks related to code classification, code retrieval, and code summarization.  





#### References 

[1] Hamilton, W., Ying, Z., & Leskovec, J. (2017). Inductive representation learning on large graphs. In *Advances in neural information processing systems* (pp. 1024-1034).

[2] Zhang, J., Wang, X., Zhang, H., Sun, H., Wang, K., & Liu, X. (2019, May). A novel neural source code representation based on abstract syntax tree. In *2019 IEEE/ACM 41st International Conference on Software Engineering (ICSE)* (pp. 783-794). IEEE.

[3] Mou, L., Li, G., Zhang, L., Wang, T., & Jin, Z. (2014). Convolutional neural networks over tree structures for programming language processing. *arXiv preprint arXiv:1409.5718*.

[4] Tufano, M., Watson, C., Bavota, G., Di Penta, M., White, M., & Poshyvanyk, D. (2018, May). Deep learning similarities from different representations of source code. In *2018 IEEE/ACM 15th International Conference on Mining Software Repositories (MSR)* (pp. 542-553). IEEE.