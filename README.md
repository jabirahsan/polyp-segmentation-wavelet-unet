# polyp-segmentation

This repository contains UNet, ResUnet and ResUnet++ models to segment colorectal polyp segmentation. The project mainly uses CVC-ClinicDB dataset. It can be modified for any segmentation data. Also, explanations from layers of the models are done. The explanation part is done by following the paper,

[LeXNet++: Layer-wise eXplainable ResUNet++ framework for segmentation of colorectal polyp cancer images] (https://link.springer.com/article/10.1007/s00521-024-10441-6)


Moreover, a custom UNet utilizing Inception Block and ASPP block is made which is named as WIUNetX. It also utilizes the explanation part. The motivation for this custom model is to correctly segment small objects and also acheive a good result in challenging situation. 

# Custom Implementation Architecture

<img width="441" height="1201" alt="InceptionUnet drawio" src="https://github.com/user-attachments/assets/5c47989f-a834-476f-a59a-fde4fdcd47ac" />


---

The encoder block  is,
<img width="821" height="661" alt="InceptionEncoder drawio (2)" src="https://github.com/user-attachments/assets/60f05ec0-2ecd-41de-b177-db0629684ef8" />

--- 
The decoder block is,


<img width="821" height="661" alt="InceptionDecoder drawio (1)" src="https://github.com/user-attachments/assets/23754ad2-3db9-4b90-9ed8-7be2e90aa14c" />

---

ASPP Block,
<img width="931" height="551" alt="ASPP drawio" src="https://github.com/user-attachments/assets/66687f83-c535-4a3e-abfa-e5312c3fee76" />


The predictions of different models are,
<img width="598" height="479" alt="image" src="https://github.com/user-attachments/assets/8b256731-5ab4-4c7e-b4bf-2859405b403c" />

The first one is the real image, second one is the ground truth, third is the UNet output, fourth is the ResUnet++ output and final is ours custom implementation output.








