# ADG-Seq2Seq: Embedding API Dependency Graph for neural code generation
### An Encoder_Decoder_Embedder based model 

Our paper:[Embedding API Dependency Graph for neural code generation](https://link.springer.com/article/10.1007/s10664-021-09968-2)

---

### Requirements
* Python 3.7
* Pytorch 1.5.0
* CUDA 1.5.0

---

### Quick Start
#### Step 1: Train a new model
To train a new model, you can run file `training.py` like:  

```
    python3 training.py [HS/MTG/EJDT]
```  
the models will be stored into `/model/*`.   
> * Since large files are not easy to upload and download on github, we put the complete data set on google drive, click [here](https://drive.google.com/drive/folders/1I3oZWdeeKT9dI4eLZmYyiJB_cXFCB0ha?usp=sharing
) for downloading. We also put some portions of dataset as demos, so before running, please rename them (if use).  
> * Before training, please new a folder `./model` to store models.


#### Step 2: Evaluation
You can run the model by:
```
    python3 eval.py [HS/MTG/EJDT]
```
---
### Examples
Code Description:
```
Reads the contents of the specified block into the buffer's page. If the buffer was dirty, then the contents of the previous page are first written to disk.
```
Example Code:  
```
void assignToBlock(BlockId blk){
    internalLock.writeLock().lock();
    try {
        this.blk=blk;
        contents.read(blk);
        lastLsn=readPage(contents, LAST_LSN_OFFSET);
    }
    finally {
        internalLock.writeLock().unlock();
    }
}
```

---
**Citation**:  
If you find this code useful in your research, please cite our paper:
```
@article{lyu2021embedding,
  title={Embedding API dependency graph for neural code generation},
  author={Lyu, Chen and Wang, Ruyun and Zhang, Hongyu and Zhang, Hanwen and Hu, Songlin},
  journal={Empirical Software Engineering},
  volume={26},
  number={4},
  pages={1--51},
  year={2021},
  publisher={Springer}
}
```

