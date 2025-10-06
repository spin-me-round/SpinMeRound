<div align="center">
<h1>[ICCV 2025] SpinMeRound: Consistent Multi-View Identity Generation Using Diffusion Models</h1>

<a href="https://arxiv.org/pdf/2504.10716"><img src="https://img.shields.io/badge/Paper-SpMR" alt="Paper PDF"></a>
<a href="https://arxiv.org/abs/2504.10716"><img src="https://img.shields.io/badge/arXiv-2503.11651-b31b1b" alt="arXiv"></a>
<a href="https://spin-me-round.github.io/"><img src="https://img.shields.io/badge/Project_Page-green" alt="Project Page"></a>


[Stathis Galanakis](https://stathisgln.github.io/), [Alexandros Lattas](https://alexlattas.com/), [Stylianos Moschoglou](https://moschoglou.com/), [Bernhard Kainz](https://bernhard-kainz.com/), [Stefanos Zafeiriou](https://www.imperial.ac.uk/people/s.zafeiriou)
</div>

```bibtex
      @misc{galanakis2025spinmeroundconsistentmultiviewidentity,
        title={SpinMeRound: Consistent Multi-View Identity Generation Using Diffusion Models}, 
        author={Stathis Galanakis and Alexandros Lattas and Stylianos Moschoglou and Bernhard Kainz and Stefanos Zafeiriou},
        year={2025},
        eprint={2504.10716},
        archivePrefix={arXiv},
        primaryClass={cs.CV},
        url={https://arxiv.org/abs/2504.10716}, 
    }
```

## TLDR
We introduce SpinMeRound: <br>
&nbsp;&nbsp;&nbsp; 🔥 An indentity consistent multi-view diffusion model <br>
&nbsp;&nbsp;&nbsp; 🔥 It can generate consisten 360 head avatars, given an input facial image. <br>
&nbsp;&nbsp;&nbsp; 🔥 It concurrently generates the corresponding shape normals for all generated views <br>

## Quick Start
1. Clone the repository and install the dependencies
```bash
git clone https://github.com/spin-me-round/SpinMeRound.git
cd SpinMeRound
pip install -r requirements.txt
```

2. Build the panohead cropping dependencies
```bash
cd sgm/panohead_cropping/FaceBoxes
sh ./build_cpu_nms.sh
cd ../../..
```

3. Download weights from [Google Drive](https://drive.google.com/drive/folders/11JHSuMHzEZ56_bIsrEQc9NGfpHloCkXD?usp=sharing) and place them into the ``` weights ``` folder


## Acknowledgements

Thanks for the contributions of the following repos: [SVD](https://github.com/Stability-AI/generative-models), [Panohead](https://github.com/SizheAn/PanoHead), [BiSeNet](https://github.com/yakhyo/face-parsing)
