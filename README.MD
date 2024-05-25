Zastosowane modele:

1. Image-to-text: zczytywanie linii tekstu z obrazu

   - https://huggingface.co/microsoft/trocr-large-handwritten

2. Generowanie dłuższego tekstu z odczytanego tekstu z obrazu

   - https://huggingface.co/openai-community/gpt2

3. Text-to-speech na podstawie wygenerowanego tekstu
   - https://huggingface.co/parler-tts/parler_tts_mini_v0.1

Instalacja pojedynczych bibliotek:

- parler_tts

```
pip install git+https://github.com/huggingface/parler-tts.git
```

- transformers

```
pip install transformers
```

- opencv

```
pip install opencv-python
```

- sounddevice

```
pip install sounddevice
```

Wszystkie potrzebne biblioteki znajdują się w requirements.txt.

```
pip install -r requirements.txt
```
