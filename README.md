<h2 align="center"><b><h3>Dual Language Models:<br>Balancing Training Efficiency and Overfitting Resilience</h3></b></h2><br>


<p align="center">
  <b>David Samuel</b> and <b>Lucas Georges Gabriel Charpentier</b>
</p>

<p align="center">
  <i>
    University of Oslo<br>
    Language Technology Group<br>
  </i>
</p>
<br>

<p align="center">
  <a href="https://arxiv.org/abs/2410.24159"><b>Paper</b></a><br>
</p>

_______

<br>

### Abstract

This paper combines autoregressive and masked-diffusion training objectives without any architectural modifications, resulting in flexible language models that outperform single-objective models. Autoregressive modeling has been a popular approach, partly because of its training efficiency; however, that comes at the cost of sensitivity to overfitting. On the other hand, masked-diffusion models are less efficient to train while being more resilient to overfitting. In this work, we demonstrate that dual-objective training achieves the best of both worlds. To derive the optimal ratio between both objectives, we train and evaluate 50 language models under varying levels of data repetition. We show that it is optimal to combine both objectives under all evaluated settings and that the optimal ratio is similar whether targeting autoregressive or masked-diffusion downstream performance.

_______

<br>

This is the official repository for Dual Language Models.

_______

<br>

### Please cite the following publication
```bibtex
@misc{samuel2025duallanguagemodelsbalancing,
      title={Dual Language Models: Balancing Training Efficiency and Overfitting Resilience}, 
      author={David Samuel and Lucas Georges Gabriel Charpentier},
      year={2025},
      eprint={2512.14549},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2512.14549}, 
}
```
