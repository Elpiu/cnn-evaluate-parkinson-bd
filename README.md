# Progetto di Tesi

**Valutazione della degenerazione del
morbo di Parkinson mediante modello
di rete neurale convoluzionale**

Candidato: Elpidio Mazza

Relatore: Prof. Rita Francese

---

### **UNIVERSITÀ DEGLI STUDI DI SALERNO**

<img src="./images_repo/unisa-1.jpg" width="200" />

### **Dipartimento di Informatica**

**Corso di Laurea Triennale in Informatica**

---

### Dataset

    #TODO

---

### CNN

    #TODO

---

# How to run in local

Per far funzionare il progetto in locale bisogna installare tutti i moduli requisiti.

```
pip install -r requirements.txt
```

Spostarsi nella cartella 'webService' e lanciare l'app flask.

```
cd .\webService\
flask run
```

### Cambiare modello

Per cambiare il modello basta sostituire il file '.h5' presente
nella cartella 'webService/static/models'.

---

### Hosted by Heroku

[Link al sito da fare]()

---

## Run with GPU

Cambiare nel file requirments.txt

```
from ---> tensorflow-cpu==2.9.1

to ---> tensorflow==2.9.1
```

se si posside una GPU, il modulo tensorflow-cpu è stato utilizzato per il deploy su Heroku.

---

# Contenuti

<div>
<p float="left">
  <img src="\images_repo\Classifier-parkinson-disease.png" width="200" /> 
  <img src="\images_repo\Classifier2.png" width="200" />
</p>
</div>
