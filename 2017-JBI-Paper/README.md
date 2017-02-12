This project is to provide the experimental framework in the paper "A Study of the Suitability of Deep Learning for Preprocessing Data in Breast Cancer Experimentation".

Folder src contains all the code to evaluate the different methods compared in the article.

Folder data provides the original immunohistochemical profile provided by the pathologists who made the original experimentation beside a manual binarization made by them, too.

Folders mortality and recidive contain ten different splits (training/test) from the original data to develop a study about the risks of overfitting of an autoencoder in this context.

Folder other contains an R file with the code to carry out a survival analyisis from the transformed databases obtained after preprocessing the original data by any preprocessing option. 