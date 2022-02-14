# UGM-PI    
    
**Paper**: Universal Generative Modeling for Calibration-free Parallel MR Imaging     
     
**Authors**: Wanqing Zhu, Bing Guan, Shanshan Wang, Minghui Zhang and Qiegen Liu      
       
          Department of Electronic Information Engineering, Nanchang University      
          Paul C. Lauterbur Research Center for Biomedical Imaging, SIAT, CAS     
   
Paper #160 accepted at IEEE ISBI 2022.          

Date : Feb-12-2022     
Version : 1.0      
The code and the algorithm are for non-comercial use only.       
Copyright 2022, Department of Electronic Information Engineering, Nanchang University.      
     
The integration of compressed sensing and parallel imaging (CS-PI) provides a robust mechanism for accelerating MRI acquisitions. However, most such strategies require the ex-plicit formation of either coil sensitivity profiles or a cross-coil correlation operator, and as a result reconstruction corresponds to solving a challenging bilinear optimization problem. In this work, we present an unsupervised deep learning framework for calibration-free parallel MRI, coined universal generative modeling for parallel imaging (UGM-PI). More precisely, we make use of the merits of both wavelet transform and the adaptive iteration strategy in a unified framework. We train a powerful noise condi-tional score network by forming wavelet tensor as the net-work input at the training phase. Experimental results on both physical phantom and in vivo datasets implied that the proposed method is comparable and even superior to state-of-the-art CS-PI reconstruction approaches.         
       
## Flowchart illustration of UGM-PI


<div align="center"><img src="https://github.com/yqx7150/UGM-PI/blob/main/Flowchart_UGM-PI.png"> </div>
<div align="center">Top: Network training process in single coil image for prior learning. Bottom: An intermediate result of the iterative process at multi-coil MRI reconstruction phase. </div>

## Convergence tendency comparison


<div align="center"><img src="https://github.com/yqx7150/UGM-PI/blob/main/Convergence.jpg"> </div>
<div align="center">Convergence tendency comparison of DSM in the native NCSN and the advanced UGM-PI, respectively. </div>

## Reconstruction comparison 1


<div align="center"><img src="https://github.com/yqx7150/UGM-PI/blob/main/Comparison-Fig3.png"> </div>
<div align="center">The test comparison of 4 coils physical phantom obtained by P-LORAKS, SAKE, LINDBREG, and the proposed UGM-PI under 2D random sampling with the acceleration factor R=6. </div>

## Reconstruction comparison 2


<div align="center"><img src="https://github.com/yqx7150/UGM-PI/blob/main/Comparison-Fig4.png"> </div>
<div align="center">Reconstruction comparison of the in vivo MPRAGE brain image obtained by P-LORAKS, SAKE, LINDBREG, and UGM-PI under 2D variable density Poisson sampling with the acceleration factor R=10. </div>

### Other Related Projects

  * Homotopic Gradients of Generative Density Priors for MR Image Reconstruction  
[<font size=5>**[Paper]**</font>](https://ieeexplore.ieee.org/abstract/document/9435335)   [<font size=5>**[Code]**</font>](https://github.com/yqx7150/HGGDP)   [<font size=5>**[PPT]**</font>](https://github.com/yqx7150/HGGDP/tree/master/Slide)

  * Iterative Reconstruction for Low-Dose CT using Deep Gradient Priors of Generative Model  
[<font size=5>**[Paper]**</font>](https://arxiv.org/abs/2009.12760)   [<font size=5>**[Code]**</font>](https://github.com/yqx7150/EASEL)   [<font size=5>**[PPT]**</font>](https://github.com/yqx7150/HGGDP/tree/master/Slide)

* Progressive Colorization via Interative Generative Models  
[<font size=5>**[Paper]**</font>](https://ieeexplore.ieee.org/document/9258392)   [<font size=5>**[Code]**</font>](https://github.com/yqx7150/iGM)   [<font size=5>**[PPT]**</font>](https://github.com/yqx7150/HGGDP/tree/master/Slide)
 
 * Joint Intensity-Gradient Guided Generative Modeling for Colorization
[<font size=5>**[Paper]**</font>](https://arxiv.org/abs/2012.14130)   [<font size=5>**[Code]**</font>](https://github.com/yqx7150/JGM)   [<font size=5>**[PPT]**</font>](https://github.com/yqx7150/HGGDP/tree/master/Slide)  [<font size=5>**[数学图像联盟会议交流PPT]**</font>](https://github.com/yqx7150/EDAEPRec/tree/master/Slide)

 * Wavelet Transform-assisted Adaptive Generative Modeling for Colorization
[<font size=5>**[Paper]**</font>](https://arxiv.org/abs/2107.04261)   [<font size=5>**[Code]**</font>](https://github.com/yqx7150/WACM)   [<font size=5>**[数学图像联盟会议交流PPT]**</font>](https://github.com/yqx7150/EDAEPRec/tree/master/Slide)


