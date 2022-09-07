# Progetto di Tesi

**Valutazione della degenerazione del
morbo di Parkinson mediante modello
di rete neurale convoluzionale**

Candidato:
* Elpidio Mazza

Relatore: 
* Prof. Rita Francese
* Dott.ssa Maria Frasca
---

### **UNIVERSITÀ DEGLI STUDI DI SALERNO**

<img src="./images_repo/unisa-1.jpg" width="200" />

### **Dipartimento di Informatica**

**Corso di Laurea Triennale in Informatica**

---

### Dataset

Numero di esempi per il training: 8740
NUmero di esempi per il test: 2210
Ogni immagine è di dimensione (150, 150)

![](images_repo/Figure_last_divisone_4_classi_numero_esempi.png)


---

### CNN

![](images_repo/modelSummary.jpg)


---

# How to run in local

Per far funzionare il progetto in locale bisogna installare tutti i moduli requisiti.

```
pip install -r requirements.txt
```


### Hosted by Heroku

[Link al sito](https://app-cnn-flask.herokuapp.com/)

---

## Run with GPU and requirments

Cambiare nel file requirments.txt

```
from ---> tensorflow-cpu==2.9.1

to ---> tensorflow==2.9.1
```

Se si posside una GPU, il modulo tensorflow-cpu è stato utilizzato per il deploy su Heroku.

Alcuni requirments sono stati rimossi per diminuire il peso della build.

---

# Contenuti

<div>
<p float="left">
  <img src=".\images_repo\Classifier-parkinson-disease.png" width="200" /> 
  <img src=".\images_repo\Classifier2.png" width="200" />
</p>
</div>
